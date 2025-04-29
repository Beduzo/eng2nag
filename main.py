from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load from Hugging Face Hub
MODEL_NAME = "xerces101/eng2nag"

tokenizer = AutoTokenizer.from_pretrained(eng2nag)
model = AutoModelForSeq2SeqLM.from_pretrained(eng2nag)

app = FastAPI()

class TranslationInput(BaseModel):
    input_text: str

@app.post("/translate/")
def translate_text(data: TranslationInput):
    inputs = tokenizer(data.input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output_text": output_text}
