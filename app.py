from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import spacy
import threading
import re
from typing import Dict, List, Tuple, Any, Set, Optional

import os
import time
import psutil
import json
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from importlib.metadata import version as pkg_version, PackageNotFoundError

app = FastAPI(title="Eartext NER")
START_TIME = time.time()

# ===== Model cache (thread-safe) =====
_models: Dict[str, "spacy.Language"] = {}
_lock = threading.Lock()

# === spaCy model per language (must be installed in the environment) ===
MODEL_BY_LANG: Dict[str, str] = {
    "es": "es_core_news_lg",
    "sv": "sv_core_news_lg",
    "da": "da_core_news_lg",
    "fi": "fi_core_news_lg",
    "en": "en_core_web_lg",
    "nl": "nl_core_news_lg",
    "pl": "pl_core_news_sm",
    "pt": "pt_core_news_lg",
}
SUPPORTED_LANGS = set(MODEL_BY_LANG.keys())

# === Config remoto de regex
CONFIG_URL = os.getenv(
    "REGEX_CONFIG_URL",
    "https://media.isaacbaltanas.com/eartext/admin/regex/config.php"
)
CONFIG_TOKEN = os.getenv("REGEX_API_TOKEN", "")
REGEX_TTL_SEC = int(os.getenv("REGEX_TTL_SEC", "60"))

def _read_int(path: str) -> int:
    try:
        with open(path, "r") as f:
            raw = f.read().strip()
        if raw.lower() == "max":
            return -1
        return int(raw)
    except Exception:
        return -2

def container_memory_info():
    lim = _read_int("/sys/fs/cgroup/memory.max")
    use = _read_int("/sys/fs/cgroup/memory.current")
    if lim != -2 and use != -2:
        if lim > 0:
            limit_mb = round(lim / (1024**2), 1)
            usage_mb = round(use / (1024**2), 1)
            percent = round((use / lim) * 100, 1) if lim else None
            return {"limit_mb": limit_mb, "usage_mb": usage_mb, "percent": percent}
    lim = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    use = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if lim != -2 and use != -2:
        if lim > 0 and lim < 1 << 60:
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
    try:
        meta = getattr(nlp, "meta", {}) or {}
        pkg_name = meta.get("name")
        version = meta.get("version") or ""
        lang_code = getattr(nlp, "lang", None) or lang
        if pkg_name and lang_code:
            return f"{lang_code}_{pkg_name}", version
    except Exception:
        pass
    return MODEL_BY_LANG.get(lang, f"{lang}_unknown"), ""

class NerRequest(BaseModel):
    text: str
    lang: str

def cut_excerpt(txt: str, start: int, end: int, win: int = 100) -> Tuple[str, str, str]:
    s = max(0, start - win)
    e = min(len(txt), end + win)
    return txt[s:start], txt[start:end], txt[end:e]

NAMED_ENTITY_LABELS = (
    "PERSON","ORG","GPE","LOC","NORP","FAC","WORK_OF_ART","EVENT",
    "PRODUCT","LANGUAGE","PER","PRS","TME","MSR","EVN","WRK","OBJ"
)

# === Normalizador de tipos venidos del Regex Manager ===
def normalize_regex_type(rtype: str) -> str:
    t = (rtype or "").strip().upper()

    # Nombres propios
    if t in {"NAME", "PROPER_NAME", "PROPN", "PERSON"}:
        return "PERSON"

    # Mapeos a etiquetas spaCy
    if t in {"GPE"}:
        return "LOC"
    if t in {"ORGANIZATION"}:
        return "ORG"
    if t in {"WORK", "WORK_OF_ART", "WRK"}:
        return "WORK_OF_ART"
    if t in {"EVN"}:
        return "EVENT"

    # Tipos numéricos / temporales
    if t in {"DATE", "TIME", "MONEY", "PERCENT", "NUMBER"}:
        return t

    # Desconocido -> NUMBER (seguro)
    return "NUMBER"

# ===== Caché de regex dinámica (con IDs) =====
# Estructura compilada: { lang: [ (compiled_regex, TYPE, rule_id), ... ] }
_regex_cache_by_lang: Dict[str, List[Tuple[re.Pattern, str, int]]] = {}
_regex_last_fetch = 0.0
_regex_lock = threading.Lock()
_regex_source = "none"  # "config" si viene del PHP; "none" si no hay reglas (sin fallback)

def _flags_from_string(fs: str) -> int:
    fs = (fs or "").lower()
    f = 0
    if "i" in fs: f |= re.IGNORECASE
    if "m" in fs: f |= re.MULTILINE
    if "s" in fs: f |= re.DOTALL
    if "x" in fs: f |= re.VERBOSE
    return f

def _compile_from_payload(payload: dict) -> Dict[str, List[Tuple[re.Pattern, str, int]]]:
    out: Dict[str, List[Tuple[re.Pattern, str, int]]] = {}
    for lang, by_type in (payload or {}).items():
        tmp: List[Tuple[re.Pattern, str, int, int]] = []
        if not isinstance(by_type, dict):
            continue
        for typ, rules in by_type.items():
            if not isinstance(rules, list):
                continue
            for r in rules:
                if not r or not r.get("enabled", True):
                    continue
                pat   = r.get("pattern", "")
                flags = _flags_from_string(r.get("flags", ""))
                prio  = int(r.get("priority", 2))
                rid   = int(r.get("id", 0))  # <<-- conservamos ID
                try:
                    rx = re.compile(pat, flags)
                    tmp.append((rx, str(typ).upper(), rid, prio))
                except re.error:
                    continue
        tmp.sort(key=lambda t: t[3])  # por prioridad
        out[lang.lower()] = [(rx, typ, rid) for (rx, typ, rid, _p) in tmp]
    return out

def _fetch_and_compile_regex():
    global _regex_cache_by_lang, _regex_last_fetch, _regex_source
    url = CONFIG_URL
    if CONFIG_TOKEN:
        sep = "&" if ("?" in url) else "?"
        url = f"{url}{sep}token={CONFIG_TOKEN}"
    req = Request(url, headers={"User-Agent": "Eartext-NER/1.0"})
    with urlopen(req, timeout=6) as r:
        data = r.read()
    payload = json.loads(data.decode("utf-8", errors="replace"))
    compiled = _compile_from_payload(payload)

    # Marca el fetch para no martillear el endpoint
    _regex_last_fetch = time.time()

    if compiled:
        _regex_cache_by_lang = compiled
        _regex_source = "config"
    else:
        # Respuesta válida pero sin reglas -> spaCy only
        _regex_cache_by_lang = {}
        _regex_source = "none"

def _ensure_regex_loaded():
    global _regex_cache_by_lang, _regex_last_fetch, _regex_source
    with _regex_lock:
        need = (time.time() - _regex_last_fetch) > REGEX_TTL_SEC
        if not _regex_cache_by_lang or need:
            try:
                _fetch_and_compile_regex()
            except (URLError, HTTPError, ValueError, json.JSONDecodeError):
                # Sin fallback: cache vacío, fuente "none", y ponemos last_fetch ahora.
                _regex_cache_by_lang = {}
                _regex_last_fetch = time.time()
                _regex_source = "none"

def get_compiled_regex(lang: str) -> Tuple[List[Tuple[re.Pattern, str, int]], str, int]:
    """
    Devuelve (lista_compilada, source, available_count)
    source = "config" | "none"
    """
    _ensure_regex_loaded()
    lang = (lang or "").lower()
    lst = _regex_cache_by_lang.get(lang, [])
    return lst, _regex_source, len(lst)

# ---------- Utilidades de origen ----------
def _format_source_tag(src_spacy: bool, regex_ids: Set[int]) -> str:
    """
    Compacta el origen para columna estrecha:
      - Solo spaCy  -> 'S'
      - Solo Regex  -> 'R#n' (o 'R#n+K' si varias)
      - Ambos       -> 'S+R#n'  **(siempre con el id de regex más bajo)**
    """
    has_r = bool(regex_ids)
    if src_spacy and has_r:
        ids_sorted = sorted([i for i in regex_ids if i > 0]) or [0]
        return f"S+R#{ids_sorted[0]}"
    if src_spacy:
        return "S"
    if has_r:
        ids_sorted = sorted([i for i in regex_ids if i > 0]) or [0]
        first = ids_sorted[0]
        rest = len(ids_sorted) - 1
        return f"R#{first}" if rest == 0 else f"R#{first}+{rest}"
    return ""

@app.post("/ner")
def ner(
    req: NerRequest,
    include_words: bool = Query(True, description="Use spaCy NER (PERSON, ORG, LOC, etc.)"),
    include_regex: bool = Query(True, description="Use Regex Manager rules (dates, numbers, names, etc.)"),
    include_numbers: bool = Query(None, description="DEPRECATED: use include_regex"),
    include_all: bool = Query(False, description="Include all occurrences per term"),
):
    txt = (req.text or "").strip()
    if not txt:
        raise HTTPException(400, "Empty text")

    lang = (req.lang or "").strip().lower()
    nlp = get_model(lang)
    model_full_name, model_version = resolved_model_name(nlp, lang)
    doc = nlp(txt)

    # Back-compat con clientes antiguos:
    do_regex = include_regex if include_numbers is None else bool(include_numbers)

    spans: List[Dict[str, Any]] = []

    PRIORITY = {
        "MONEY": 3, "DATE": 3, "TIME": 3, "PERCENT": 3,
        "PERSON": 2, "ORG": 2, "LOC": 2, "WORK_OF_ART": 2, "EVENT": 2,
        "NUMBER": 1
    }
    def _prio(t: str) -> int:
        return PRIORITY.get(t, 2)



    def _merge_sources(dst: Dict[str, Any], src_spacy: bool, src_regex_id: Optional[int]):
        if src_spacy:
            dst["src_spacy"] = True
        if src_regex_id is not None:
            dst["src_regex_ids"].add(int(src_regex_id))

    def add_span(start: int, end: int, typ: str, text: str, *, src_spacy: bool = False, src_regex_id: Optional[int] = None):
        # Estructura extendida con fuentes
        new_span = {"text": text, "type": typ, "start": start, "end": end, "src_spacy": False, "src_regex_ids": set()}  # type: ignore
        _merge_sources(new_span, src_spacy, src_regex_id)

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
                if p_new > p_old: replace = True
                elif p_new == p_old and (end - start) > (d - c): replace = True

                if replace:
                    # Fusiona origen antes de reemplazar
                    _merge_sources(new_span, s.get("src_spacy", False), None)
                    for rid in s.get("src_regex_ids", set()):
                        _merge_sources(new_span, False, rid)
                    spans.pop(i)
                    continue
                else:
                    # Conserva s y añade el origen del nuevo span a s
                    _merge_sources(s, src_spacy, src_regex_id)
                    return
            i += 1

        spans.append(new_span)

    # ---- NER entities (spaCy) ----
    if include_words:
        for ent in doc.ents:
            if ent.label_ in NAMED_ENTITY_LABELS:
                label = ent.label_
                if label == "GPE": label = "LOC"
                if label in ("PER", "PRS"): label = "PERSON"
                if label == "WRK": label = "WORK_OF_ART"
                if label == "EVN": label = "EVENT"
                add_span(ent.start_char, ent.end_char, label, ent.text, src_spacy=True)

    # ---- Regex entities (con IDs y metadatos) ----
    used_rule_ids: Set[int] = set()
    regex_source = "none"
    available_rules = 0

    if do_regex:
        compiled_list, regex_source, available_rules = get_compiled_regex(lang)
        for rx, rtype_raw, rid in compiled_list:
            out_type = normalize_regex_type(rtype_raw)
            for m in rx.finditer(txt):
                val = m.group(0)

                # Higiene solo para tipos numéricos
                if out_type in ("NUMBER", "MONEY", "PERCENT"):
                    stripped = re.sub(r"\s*(€|eur|euros?|kr|sek|dkk|zł|pln|pln\.?)\s*$", "", val, flags=re.IGNORECASE)
                    stripped = re.sub(r"\s*%\s*$", "", stripped)
                    core = stripped.replace("\u00A0", " ")
                    core = core.replace(" ", "").replace(".", "").replace(",", "")
                    if re.fullmatch(r"0+", core or ""):
                        continue

                add_span(m.start(), m.end(), out_type, val, src_regex_id=rid)
                if rid:
                    used_rule_ids.add(int(rid))

    # ---- Agrupar por (text, type) ----
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
                "occurrences": [],
                "src_spacy": False,
                "src_regex_ids": set(),  # set[int]
            }
        groups[key]["count"] += 1
        # Propaga origen al grupo
        if s.get("src_spacy", False):
            groups[key]["src_spacy"] = True
        for rid in s.get("src_regex_ids", set()):
            groups[key]["src_regex_ids"].add(int(rid))

    if include_all:
        for s in spans:
            pre, mid, post = cut_excerpt(txt, s["start"], s["end"], win=100)
            groups[(s["text"], s["type"])]["occurrences"].append({
                "start": s["start"], "end": s["end"], "pre": pre, "match": mid, "post": post
            })

    # Construye summary con tag de origen compacto
    summary = []
    for g in groups.values():
        source_tag = _format_source_tag(bool(g["src_spacy"]), set(g["src_regex_ids"]))
        summary.append({
            "text": g["text"],
            "type": g["type"],
            "count": g["count"],
            "source": source_tag,                 # 'S', 'R#n'/'R#n+K' o 'S+R#n'
            "excerpt": g["first_excerpt"]
        })
    summary.sort(key=lambda x: x["count"], reverse=True)

    occurrences = []
    if include_all:
        for g in groups.values():
            occurrences.append({
                "text": g["text"],
                "type": g["type"],
                "occurrences": g["occurrences"]
            })

    # --- Bloque de metadatos de regex ---
    regex_meta = {
        "source": regex_source,  # "config" -> Curated ; "none" -> no regex rules
        "used_rule_ids": sorted(list(used_rule_ids)),
        "used_rule_count": len(used_rule_ids),
        "available_rule_count": int(available_rules),
    }

    return {
        "lang": lang,
        "model": model_full_name,
        "model_version": model_version,
        "regex": regex_meta,
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

@app.post("/regex/reload")
def reload_regex():
    global _regex_last_fetch
    with _regex_lock:
        _regex_last_fetch = 0.0
    return {"ok": True, "message": "Regex cache will refresh on next request"}

@app.get("/status")
def status():
    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cgroup_mem = container_memory_info()
    if cgroup_mem:
        mem_total_mb = cgroup_mem["limit_mb"]
        mem_used_mb  = cgroup_mem["usage_mb"]
        mem_percent  = cgroup_mem["percent"]
    else:
        mem_total_mb = round(vm.total / (1024**2), 1)
        mem_used_mb  = round((vm.total - vm.available) / (1024**2), 1)
        mem_percent  = vm.percent

    proc = psutil.Process(os.getpid())
    with proc.oneshot():
        rss_bytes = proc.memory_info().rss
        threads = proc.num_threads()
    uptime_sec = int(time.time() - START_TIME)

    loaded = []
    with _lock:
        for lang_code, nlp in _models.items():
            name, ver = resolved_model_name(nlp, lang_code)
            loaded.append({"lang": lang_code, "model": name, "version": ver})
        loaded_count = len(_models)

    installed = {}
    for code, pkg in MODEL_BY_LANG.items():
        try:
            ver = pkg_version(pkg)
            installed[code] = {"package": pkg, "installed": True, "version": ver}
        except PackageNotFoundError:
            installed[code] = {"package": pkg, "installed": False, "version": None}

    regex_cache_state = {
        "config_url": CONFIG_URL,
        "ttl_sec": REGEX_TTL_SEC,
        "last_fetch_age_sec": None if _regex_last_fetch == 0 else int(time.time() - _regex_last_fetch),
        "langs_cached": sorted(list(_regex_cache_by_lang.keys())),
        "source": _regex_source,  # "config" or "none"
    }

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
            "mapping": MODEL_BY_LANG,
            "installed": installed,
            "loaded_count": loaded_count,
            "loaded": loaded,
        },
        "regex_cache": regex_cache_state,
    }
