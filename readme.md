# Eartext NER

Render deploy for Eartext NER microservice (FastAPI + spaCy).  
Uses small spaCy models (`_sm`) for testing (sv, da, fi, en, pl, fr, es, it, tr, nl, de).  
Endpoint: `/ner`  
Request: `{"text": "...", "lang": "sv"}`  
