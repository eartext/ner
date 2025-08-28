from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import spacy
import threading
import re
from typing import Dict, List, Tuple

app = FastAPI(title="Eartext NER (sv/es/pl)")

# ===== Cache de modelos =====
_models: Dict[str, "spacy.Language"] = {}
_lock = threading.Lock()

MODEL_BY_LANG = {
    "sv": "sv_core_news_lg",   # Sueco grande
    "es": "es_core_news_lg",   # Español grande
    "pl": "pl_core_news_sm",   # Polaco (no hay lg)
}
# Idiomas aceptados en la petición: "sv" | "es" | "pl"

def get_model(lang: str):
    lang = (lang or "").strip().lower()
    if lang not in MODEL_BY_LANG:
        raise HTTPException(400, f"Unsupported lang: {lang}")
    with _lock:
        if lang not in _models:
            nlp = spacy.load(MODEL_BY_LANG[lang], disable=["lemmatizer", "textcat"])
            if "ner" not in nlp.pipe_names:
                raise HTTPException(500, f"NER pipe not available for {lang}")
            _models[lang] = nlp
    return _models[lang]

class NerRequest(BaseModel):
    text: str
    lang: str  # "sv" | "es" | "pl"

def cut_excerpt(txt: str, start: int, end: int, win: int = 100) -> Tuple[str, str, str]:
    s = max(0, start - win)
    e = min(len(txt), end + win)
    pre  = txt[s:start]
    mid  = txt[start:end]
    post = txt[end:e]
    return pre, mid, post

# ===== Etiquetas de entidades que sí mostramos =====
# Normalizamos algunas (GPE->LOC, PER/PRS->PERSON, WRK->WORK_OF_ART, EVN->EVENT).
NAMED_ENTITY_LABELS = (
    "PERSON","ORG","GPE","LOC","NORP","FAC","WORK_OF_ART","EVENT",
    "PRODUCT","LANGUAGE","PER","PRS","TME","MSR","EVN","WRK","OBJ"
)

# ===== Regex por idioma (con tipo: DATE, TIME, MONEY, PERCENT, NUMBER) =====
REGEX_BY_LANG = {
    "es": [
        (re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"), "DATE"),                              # 23/09/2024, 01-01-2025, 15.03.2026
        (re.compile(r"\b\d{1,2}\s+de\s+[A-Za-záéíóúñ]+\s+\d{4}\b", re.IGNORECASE), "DATE"),       # 14 de febrero de 2021
        (re.compile(r"\b\d{1,2}:\d{2}\b"), "TIME"),                                               # 08:30, 17:45
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\s*(?:€|eur|euros?)", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:€|eur|euros?)", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*%"), "PERCENT"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\b"), "NUMBER"),                    # 1.234.567,89 | 1.234.567.89
        (re.compile(r"\b\d+(?:[.,]\d+)?\b"), "NUMBER"),                                           # 123 | 3,7 | 6.0
    ],
    "sv": [
        (re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}\s+[A-Za-zåäöÅÄÖ]+\s+\d{4}\b"), "DATE"),                           # 14 februari 2021
        (re.compile(r"\b\d{1,2}:\d{2}\b"), "TIME"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\s*(?:kr|SEK)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:kr|SEK)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*%"), "PERCENT"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\b"), "NUMBER"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\b"), "NUMBER"),
    ],
    "da": [
        (re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}\.\s+[A-Za-zæøåÆØÅ]+\s+\d{4}\b"), "DATE"),                         # 14. februar 2021
        (re.compile(r"\b\d{1,2}:\d{2}\b"), "TIME"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\s*(?:kr|DKK)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:kr|DKK)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*%"), "PERCENT"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\b"), "NUMBER"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\b"), "NUMBER"),
    ],
    "fi": [
        (re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"), "DATE"),                                   # 23.09.2024
        (re.compile(r"\b\d{1,2}\.\s+[A-Za-zäöÄÖ]+\s+\d{4}\b"), "DATE"),                           # 14. helmikuuta 2021
        (re.compile(r"\b\d{1,2}:\d{2}\b"), "TIME"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\s*€"), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*€"), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*%"), "PERCENT"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\b"), "NUMBER"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\b"), "NUMBER"),
    ],
    "pl": [
        (re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"), "DATE"),
        (re.compile(r"\b\d{1,2}\s+[A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ]+\s+\d{4}\b"), "DATE"),               # 14 lutego 2021
        (re.compile(r"\b\d{1,2}:\d{2}\b"), "TIME"),
        (re.compile(r"\b\d{1,3}(?:\.(?:\s)?\d{3})+(?:[.,]\d+)?\s*(?:zł|PLN)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:zł|PLN)\b", re.IGNORECASE), "MONEY"),
        (re.compile(r"\b\d+(?:[.,]\d+)?\s*%"), "PERCENT"),
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

    nlp = get_model(req.lang)
    doc = nlp(txt)

    spans: List[Dict] = []

    # --- util para añadir spans con resolución de solapamientos ---
    PRIORITY = {"MONEY": 3, "DATE": 3, "TIME": 3, "PERCENT": 3, "NUMBER": 1}
    # lo no numérico del NER (PERSON/ORG/LOC...) les damos prioridad 2
    def _prio(t: str) -> int:
        return PRIORITY.get(t, 2)

    def add_span(start: int, end: int, typ: str, text: str):
        """Inserta un span resolviendo solapamientos:
           - Si solapa con otro y tiene mayor prioridad, reemplaza al existente.
           - Si misma prioridad y este es más largo, reemplaza al existente.
           - Si peor, se descarta.
        """
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
                    # no incrementamos i; reevaluamos contra los restantes
                    continue
                else:
                    # el existente gana: no añadimos este
                    return
            i += 1

        # si llega aquí, no hubo bloqueos => añadimos
        spans.append({"text": text, "type": typ, "start": start, "end": end})

    # ---- Entidades del modelo (personas, orgs, etc.) ----
    if include_words:
        for ent in doc.ents:
            if ent.label_ in NAMED_ENTITY_LABELS:
                label = ent.label_
                # Normalizaciones entre modelos
                if label == "GPE":
                    label = "LOC"
                if label in ("PER", "PRS"):
                    label = "PERSON"
                if label == "WRK":
                    label = "WORK_OF_ART"
                if label == "EVN":
                    label = "EVENT"
                add_span(ent.start_char, ent.end_char, label, ent.text)

    # ---- Números/fechas/porcentajes/divisas por regex ----
    if include_numbers:
        for regex, rtype in REGEX_BY_LANG.get(req.lang, []):
            for m in regex.finditer(txt):
                val = m.group(0)

                # Evitar “000”, “000.000”, “0 000,00”, etc. (solo ceros y separadores)
                if rtype in ("NUMBER", "MONEY", "PERCENT") and re.fullmatch(r"[0\s]+(?:[.,][0\s]+)?", val):
                    continue

                # Para la salida, mantenemos el tipo específico si lo hay; si no, NUMBER
                out_type = rtype if rtype in ("DATE", "TIME", "MONEY", "PERCENT") else "NUMBER"
                add_span(m.start(), m.end(), out_type, val)

    # ---- Agrupación por (texto, tipo) ----
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
        "lang": req.lang.lower(),
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
