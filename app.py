# app.py
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer
import uvicorn

# ۱. د ماډل جوړښت (Architecture)
class GhanamModel(torch.nn.Module):
    def __init__(self, vocab_size=65536, n_embd=768, n_head=12, n_layer=12):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, n_embd)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head, dim_feedforward=3072, batch_first=True
            ) for _ in range(n_layer)
        ])
        self.norm_final = torch.nn.LayerNorm(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        x = self.embeddings(idx)
        for layer in self.layers: x = layer(x)
        return self.lm_head(self.norm_final(x))

# ۲. د تنظیماتو لوډ کول
app = FastAPI(title="Ghanam-1B Pashto AI Backend")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file("Ghanam-1B-Tokenizer_Fixed.json")
model = GhanamModel(vocab_size=65536)

# د وزنونو بارول
checkpoint = torch.load("ghanam_checkpoint_step_20000.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

class ChatRequest(BaseModel):
    prompt: str
    max_len: int = 50
    temp: float = 0.7

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        input_ids = torch.tensor([tokenizer.encode(request.prompt).ids]).to(device)
        generated = input_ids
        
        for _ in range(request.max_len):
            with torch.no_grad():
                logits = model(generated)[:, -1, :] / request.temp
                # د تکرار ضد جریمه
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= 1.4
                
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                if next_token.item() == tokenizer.token_to_id("</s>"): break
        
        reply = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        return {"status": "success", "response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)