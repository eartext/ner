from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import threading
import re

app = FastAPI(title="Eartext NER")

# Carga perezosa de modelos
_models = {}
_lock = threading.Lock()

# Normalizaciones de códigos
ALIASES = {
    "es-es": "es", "es-la": "es", "es-mx": "es", "es-ar": "es",
}

# Modelos pequeños para pruebas
MODEL_BY_LANG = {
    "sv": "sv_core_news_sm",  # Sueco
    "da": "da_core_news_sm",  # Danés
    "fi": "fi_core_news_sm",  # Finés
    "en": "en_core_web_sm",   # Inglés
    "pl": "pl_core_news_sm",  # Polaco
    "fr": "fr_core_news_sm",  # Francés
    "es": "es_core_news_sm",  # Español (sirve para ES/LatAm)
    "it": "it_core_news_sm",  # Italiano
    "tr": "tr_core_news_sm",  # Turco
    "nl": "nl_core_news_sm",  # Neerlandés
    "de": "de_core_news_sm",  # Alemán
}

def normalize_lang(lang: str) -> str:
    key = (lang or "").strip().lower()
    return ALIASES.get(key, key)

def get_model(lang: str):
    lang = normalize_lang(lang)
    if lang not in MODEL_BY_LANG:
        raise HTTPException(400, f"Unsupported lang: {lang}")
    with _lock:
        if lang not in _models:
            nlp = spacy.load(MODEL_BY_LANG[lang], disable=["lemmatizer","textcat"])
            if "ner" not in nlp.pipe_names:
                raise HTTPException(500, f"NER pipe not available for {lang}")
            _models[lang] = nlp
    return _models[lang]

class NerRequest(BaseModel):
    text: str
    lang: str

@app.post("/ner")
def ner(req: NerRequest):
    txt = (req.text or "").strip()
    if not txt:
        raise HTTPException(400, "Empty text")
    nlp = get_model(req.lang)
    doc = nlp(txt)

    entities = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON","ORG","GPE","LOC","NORP","FAC","WORK_OF_ART","EVENT"):
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

    # Detecta números manualmente
    for m in re.finditer(r"\b\d+(?:[.,]\d+)?\b", txt):
        entities.append({
            "text": m.group(0),
            "type": "NUMBER",
            "start": m.start(),
            "end": m.end()
        })

    # Deduplicar
    seen = set()
    dedup = []
    for e in entities:
        key = (e["text"], e["type"], e["start"], e["end"])
        if key not in seen:
            seen.add(key)
            dedup.append(e)

    return {"lang": normalize_lang(req.lang), "entities": dedup}
