from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import spacy
import threading
import re
from typing import Dict, List, Tuple

import os
import time
import psutil

from importlib.metadata import version as pkg_version, PackageNotFoundError

app = FastAPI(title="Eartext NER")
START_TIME = time.time()
# ===== Model cache (thread-safe) =====
_models: Dict[str, "spacy.Language"] = {}
_lock = threading.Lock()

# === spaCy model per language (must be installed in the environment) ===
MODEL_BY_LANG: Dict[str, str] = {
    "es": "es_core_news_lg",  # Español
    "sv": "sv_core_news_lg",  # Sueco
    "da": "da_core_news_lg",  # Danés
    "fi": "fi_core_news_lg",  # Finés
    "en": "en_core_web_lg",   # Inglés
    "nl": "nl_core_news_lg",  # Neerlandés
    "pl": "pl_core_news_sm",  # Polaco (no hay lg oficial)
    "pt": "pt_core_news_lg",  # Portugués (Brasil)
    # Si añades más, inclúyelos aquí y, si quieres, añade regex abajo.
}

SUPPORTED_LANGS = set(MODEL_BY_LANG.keys())

def _read_int(path: str) -> int:
    try:
        with open(path, "r") as f:
            raw = f.read().strip()
        if raw.lower() == "max":   # cgroup v2 “sin límite”
            return -1
        return int(raw)
    except Exception:
        return -2  # error

def container_memory_info():
    """
    Devuelve dict con {limit_mb, usage_mb, percent} leyendo cgroups.
    Soporta cgroup v2 (memory.max/current) y v1 (memory.limit_in_bytes/usage_in_bytes).
    Si no se puede determinar, devuelve None.
    """
    # cgroup v2
    lim = _read_int("/sys/fs/cgroup/memory.max")
    use = _read_int("/sys/fs/cgroup/memory.current")
    if lim != -2 and use != -2:
        if lim > 0:  # hay límite explícito
            limit_mb = round(lim / (1024**2), 1)
            usage_mb = round(use / (1024**2), 1)
            percent = round((use / lim) * 100, 1) if lim else None
            return {"limit_mb": limit_mb, "usage_mb": usage_mb, "percent": percent}
        # lim == -1 -> “max” (sin límite); seguimos intentando v1

    # cgroup v1
    lim = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    use = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if lim != -2 and use != -2:
        # algunos hosts reportan un límite gigantesco (≈ no limitado)
        if lim > 0 and lim < 1 << 60:  # descarta valores absurdamente grandes
            limit_mb = round(lim / (1024**2), 1)
            usage_mb = round(use / (1024**2), 1)
            percent = round((use / lim) * 100, 1) if lim else None
            return {"limit_mb": limit_mb, "usage_mb": usage_mb, "percent": percent}

    return None

def get_model(lang: str):
    lang = (lang or "").strip().lower()
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported lang='{lang}'. Supported: {sorted(SUPPORTED_LANGS)}"
        )
    with _lock:
        if lang not in _models:
            nlp = spacy.load(MODEL_BY_LANG[lang], disable=["lemmatizer", "textcat"])
            if "ner" not in nlp.pipe_names:
                raise HTTPException(500, f"NER pipe not available for '{lang}'")
            _models[lang] = nlp
    return _models[lang]

def resolved_model_name(nlp: "spacy.Language", lang: str) -> Tuple[str, str]:
    """
    Devuelve (model_full_name, model_version).
    Intenta construir p.ej. 'es_core_news_lg' desde meta+lang;
    si no puede, cae al nombre del mapping (MODEL_BY_LANG[lang]).
    """
    try:
        meta = getattr(nlp, "meta", {}) or {}
        pkg_name = meta.get("name")  # suele ser 'core_news_lg'
        version = meta.get("version") or ""
        lang_code = getattr(nlp, "lang", None) or lang
        if pkg_name and lang_code:
            return f"{lang_code}_{pkg_name}", version
    except Exception:
        pass
    # Fallback: lo que cargamos desde el mapping
    return MODEL_BY_LANG.get(lang, f"{lang}_unknown"), ""

class NerRequest(BaseModel):
    text: str
    lang: str  # e.g. "es" | "sv" | "da" | "fi" | "en" | "nl" | "pl"

def cut_excerpt(txt: str, start: int, end: int, win: int = 100) -> Tuple[str, str, str]:
    s = max(0, start - win)
    e = min(len(txt), end + win)
    pre  = txt[s:start]
    mid  = txt[start:end]
    post = txt[end:e]
    return pre, mid, post

# ===== Entity labels we keep (normalized across models) =====
NAMED_ENTITY_LABELS = (
    "PERSON","ORG","GPE","LOC","NORP","FAC","WORK_OF_ART","EVENT",
    "PRODUCT","LANGUAGE","PER","PRS","TME","MSR","EVN","WRK","OBJ"
)

# ===== Regex per language (DATE, TIME, MONEY, PERCENT, NUMBER) =====
REGEX_BY_LANG = {
    "es": [
        (re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}\s+de\s+[A-Za-záéíóúñ]+\s+\d{4}\b", re.IGNORECASE), "DATE"),
        (re.compile(r"\b\d{1,2}:\d{2}\b"), "TIME"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\s*(?:€|eur|euros?)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:€|eur|euros?)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*%\b"), "PERCENT"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\b"), "NUMBER"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\b"), "NUMBER"),
    ],
    "sv": [
        (re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}\s+[A-Za-zåäöÅÄÖ]+\s+\d{4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}:\d{2}\b"), "TIME"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\s*(?:kr|SEK)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:kr|SEK)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*%\b"), "PERCENT"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\b"), "NUMBER"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\b"), "NUMBER"),
    ],
    "da": [
        (re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}\.\s+[A-Za-zæøåÆØÅ]+\s+\d{4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}:\d{2}\b"), "TIME"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\s*(?:kr|DKK)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:kr|DKK)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*%\b"), "PERCENT"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\b"), "NUMBER"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\b"), "NUMBER"),
    ],
    "fi": [
        (re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}\.\s+[A-Za-zäöÄÖ]+\s+\d{4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}:\d{2}\b"), "TIME"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\s*€\b"), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*€\b"), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*%\b"), "PERCENT"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\b"), "NUMBER"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\b"), "NUMBER"),
    ],
    "pl": [
        (re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}\s+[A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ]+\s+\d{4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}:\d{2}\b"), "TIME"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\s*(?:zł|PLN)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:zł|PLN)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*%\b"), "PERCENT"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\b"), "NUMBER"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\b"), "NUMBER"),
    ],
}

@app.post("/ner")
def ner(
    req: NerRequest,
    include_words: bool = Query(True, description="Include named entities (PERSON, ORG, etc.)"),
    include_numbers: bool = Query(True, description="Include numbers/dates/amounts via regex"),
    include_all: bool = Query(False, description="Include all occurrences per term"),
):
    txt = (req.text or "").strip()
    if not txt:
        raise HTTPException(400, "Empty text")

    lang = (req.lang or "").strip().lower()
    nlp = get_model(lang)
    model_full_name, model_version = resolved_model_name(nlp, lang)
    doc = nlp(txt)

    spans: List[Dict] = []

    # --- priority for overlap resolution ---
    PRIORITY = {"MONEY": 3, "DATE": 3, "TIME": 3, "PERCENT": 3, "NUMBER": 1}
    def _prio(t: str) -> int:
        return PRIORITY.get(t, 2)  # NER words -> 2

    def add_span(start: int, end: int, typ: str, text: str):
        i = 0
        while i < len(spans):
            s = spans[i]
            a, b = start, end
            c, d = s["start"], s["end"]

            overlap = not (b <= c or a >= d)
            if overlap:
                p_new = _prio(typ)
                p_old = _prio(s["type"])

                replace = False
                if p_new > p_old:
                    replace = True
                elif p_new == p_old and (end - start) > (d - c):
                    replace = True

                if replace:
                    spans.pop(i)
                    continue
                else:
                    return
            i += 1

        spans.append({"text": text, "type": typ, "start": start, "end": end})

    # ---- NER entities ----
    if include_words:
        for ent in doc.ents:
            if ent.label_ in NAMED_ENTITY_LABELS:
                label = ent.label_
                if label == "GPE":
                    label = "LOC"
                if label in ("PER", "PRS"):
                    label = "PERSON"
                if label == "WRK":
                    label = "WORK_OF_ART"
                if label == "EVN":
                    label = "EVENT"
                add_span(ent.start_char, ent.end_char, label, ent.text)

    # ---- Numbers/dates/money/percent via regex ----
    if include_numbers:
        for regex, rtype in REGEX_BY_LANG.get(lang, []):
            for m in regex.finditer(txt):
                val = m.group(0)

                # filter pure zeros
                if rtype in ("NUMBER", "MONEY", "PERCENT"):
                    stripped = re.sub(r"\s*(€|eur|euros?|kr|sek|dkk|zł|pln|pln\.?)\s*$", "", val, flags=re.IGNORECASE)
                    stripped = re.sub(r"\s*%\s*$", "", stripped)
                    core = stripped.replace("\u00A0", " ")
                    core = core.replace(" ", "").replace(".", "").replace(",", "")
                    if re.fullmatch(r"0+", core or ""):
                        continue

                out_type = rtype if rtype in ("DATE", "TIME", "MONEY", "PERCENT") else "NUMBER"
                add_span(m.start(), m.end(), out_type, val)

    # ---- Group by (text, type) ----
    groups: Dict[Tuple[str, str], Dict] = {}
    for s in spans:
        key = (s["text"], s["type"])
        if key not in groups:
            pre, mid, post = cut_excerpt(txt, s["start"], s["end"], win=100)
            groups[key] = {
                "text": s["text"],
                "type": s["type"],
                "count": 0,
                "first_excerpt": {"pre": pre, "match": mid, "post": post},
                "occurrences": []
            }
        groups[key]["count"] += 1
        if include_all:
            pre, mid, post = cut_excerpt(txt, s["start"], s["end"], win=100)
            groups[key]["occurrences"].append({
                "start": s["start"], "end": s["end"],
                "pre": pre, "match": mid, "post": post
            })

    summary = [{
        "text": g["text"],
        "type": g["type"],
        "count": g["count"],
        "excerpt": g["first_excerpt"]
    } for g in groups.values()]
    summary.sort(key=lambda x: x["count"], reverse=True)

    occurrences = []
    if include_all:
        occurrences = [{
            "text": g["text"],
            "type": g["type"],
            "occurrences": g["occurrences"]
        } for g in groups.values()]

    return {
        "lang": lang,
        "model": model_full_name,         # <<--- NUEVO
        "model_version": model_version,   # <<--- opcional, útil para auditoría
        "summary": summary,
        "occurrences": occurrences
    }

@app.post("/reset")
def reset_models():
    global _models
    with _lock:
        cnt = len(_models)
        _models.clear()
    import gc; gc.collect()
    return {"ok": True, "cleared": cnt}

@app.get("/status")
def status():
    """
    Métricas de sistema/proceso y estado de modelos (instalados vs cargados).
    Prioriza los límites de memoria del contenedor (cgroups) si están disponibles.
    """
    # --- Sistema (usa memoria del contenedor si existe) ---
    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)

    cgroup_mem = container_memory_info()
    if cgroup_mem:
        mem_total_mb = cgroup_mem["limit_mb"]
        mem_used_mb  = cgroup_mem["usage_mb"]
        mem_percent  = cgroup_mem["percent"]
    else:
        # Fallback al host (menos preciso en PaaS)
        mem_total_mb = round(vm.total / (1024**2), 1)
        mem_used_mb  = round((vm.total - vm.available) / (1024**2), 1)
        mem_percent  = vm.percent

    # --- Proceso actual ---
    proc = psutil.Process(os.getpid())
    with proc.oneshot():
        rss_bytes = proc.memory_info().rss
        threads = proc.num_threads()
    uptime_sec = int(time.time() - START_TIME)

    # --- Modelos cargados (detallado) ---
    loaded = []
    with _lock:
        for lang_code, nlp in _models.items():
            name, ver = resolved_model_name(nlp, lang_code)
            loaded.append({"lang": lang_code, "model": name, "version": ver})
        loaded_count = len(_models)

    # --- Modelos instalados (por paquete) ---
    installed = {}
    for code, pkg in MODEL_BY_LANG.items():
        try:
            ver = pkg_version(pkg)
            installed[code] = {"package": pkg, "installed": True, "version": ver}
        except PackageNotFoundError:
            installed[code] = {"package": pkg, "installed": False, "version": None}

    return {
        "ok": True,
        "server": {
            "pid": os.getpid(),
            "uptime_sec": uptime_sec,
            "threads": threads,
            "spacy_version": spacy.__version__,
        },
        "system": {
            "cpu_percent": cpu_percent,
            "mem_total_mb": mem_total_mb,
            "mem_used_mb": mem_used_mb,
            "mem_percent": mem_percent,
            "proc_rss_mb": round(rss_bytes / (1024**2), 1),
        },
        "models": {
            "supported_langs": sorted(list(SUPPORTED_LANGS)),
            "mapping": MODEL_BY_LANG,      # lang -> paquete esperado
            "installed": installed,        # instalado o no + versión
            "loaded_count": loaded_count,  # cuántos en RAM
            "loaded": loaded,              # detalle de cargados
        },
    }
