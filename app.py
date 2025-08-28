from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import spacy
import threading
import re
from typing import Dict, List, Tuple

app = FastAPI(title="Eartext NER (sv/es/pl)")

# ===== Modelos (solo 3 idiomas para pruebas) =====
_models: Dict[str, "spacy.Language"] = {}
_lock = threading.Lock()

MODEL_BY_LANG = {
    "sv": "sv_core_news_sm",   # Sueco
    "es": "es_core_news_sm",   # Español
    "pl": "pl_core_news_sm",   # Polaco
}

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

# Etiquetas que queremos incluir como "palabras" (entidades con nombre)
NAMED_ENTITY_LABELS = (
    "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "WORK_OF_ART", "EVENT",
    "PRODUCT", "LANGUAGE", "PER", "MISC"
)

# Números con millares y decimales en formatos europeos y anglosajones
NUMBER_REGEX = re.compile(
    r"\b\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?\b|\b\d+(?:[.,]\d+)?\b"
)

@app.post("/ner")
def ner(
    req: NerRequest,
    include_words: bool = Query(True, description="Include named entities (PERSON, ORG, etc.)"),
    include_numbers: bool = Query(True, description="Include numbers"),
    include_all: bool = Query(False, description="Include all occurrences per term"),
):
    txt = (req.text or "").strip()
    if not txt:
        raise HTTPException(400, "Empty text")

    nlp = get_model(req.lang)
    doc = nlp(txt)

    # 1) Recolectar spans según interruptores
    spans: List[Dict] = []

    if include_words:
        for ent in doc.ents:
            if ent.label_ in NAMED_ENTITY_LABELS:
                label = ent.label_
                # Unificamos GPE (geo-político) en LOC (lugar), para simplificar la salida
                if label == "GPE":
                    label = "LOC"
                spans.append({
                    "text": ent.text,
                    "type": label,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

    if include_numbers:
        for m in NUMBER_REGEX.finditer(txt):
            spans.append({
                "text": m.group(0),
                "type": "NUMBER",
                "start": m.start(),
                "end": m.end()
            })

    # 2) Agregar por (texto exacto + tipo)
    groups: Dict[Tuple[str, str], Dict] = {}
    for s in spans:
        key = (s["text"], s["type"])  # si quisieras unificar mayúsculas: (s["text"].casefold(), s["type"])
        if key not in groups:
            pre, mid, post = cut_excerpt(txt, s["start"], s["end"], win=100)
            groups[key] = {
                "text": s["text"],        # conserva tal cual el primero visto
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

    # 3) Summary ordenado por frecuencia desc
    summary = [{
        "text": g["text"],
        "type": g["type"],
        "count": g["count"],
        "excerpt": g["first_excerpt"]  # {pre, match, post}
    } for g in groups.values()]
    summary.sort(key=lambda x: x["count"], reverse=True)

    # 4) Occurrences opcional (para tu futura segunda tabla)
    occurrences = []
    if include_all:
        occurrences = [{
            "text": g["text"],
            "type": g["type"],
            "occurrences": g["occurrences"]
        } for g in groups.values()]

    return {
        "lang": req.lang.lower(),
        "summary": summary,          # para la tabla principal
        "occurrences": occurrences   # solo si include_all=true
    }
