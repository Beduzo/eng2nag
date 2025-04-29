from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load your model
tokenizer = AutoTokenizer.from_pretrained("./saved model")   # Put your model folder name
model = AutoModelForSeq2SeqLM.from_pretrained("./saved model")

app = FastAPI()

# Define the input format
class TextInput(BaseModel):
    input_text: str

@app.post("/translate/")
def translate_text(data: TextInput):
    inputs = tokenizer(data.input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output_text": translated_text}
