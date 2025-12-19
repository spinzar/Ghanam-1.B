import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer
from safetensors.torch import load_model
import numpy as np

# ۱. د ماډل ارکیټیکچر (Architecture)
class GhanamModel(torch.nn.Module):
    def __init__(self, vocab_size=65536, n_embd=768, n_head=12, n_layer=12):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, n_embd)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head, dim_feedforward=3072, 
                batch_first=True, activation='gelu'
            ) for _ in range(n_layer)
        ])
        self.norm_final = torch.nn.LayerNorm(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        x = self.embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm_final(x))

# ۲. د Backend تنظیمات
app = FastAPI(title="Ghanam-1B Pashto AI Engine")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# د ټوکنر او ماډل لوډ کول
try:
    tokenizer = Tokenizer.from_file("Ghanam-1B-Tokenizer_Fixed.json")
    model = GhanamModel()
    load_model(model, "model.safetensors")
    model.to(device)
    model.eval()
    print(f"✅ Ghanam-1B is LIVE on {device}")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

# ۳. د غوښتنې سټراکچر (Input Validation)
class GhanamRequest(BaseModel):
    prompt: str
    max_len: int = 64
    temperature: float = 0.75
    top_p: float = 0.92
    top_k: int = 50
    repetition_penalty: float = 1.8  # د تکرار مخنیوي لپاره جریمه

# ۴. د تولید سمارټ منطق (Generation Logic)
def generate_response(req: GhanamRequest):
    input_ids = torch.tensor([tokenizer.encode(req.prompt).ids]).to(device)
    generated = input_ids
    
    for _ in range(req.max_len):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs[:, -1, :] / req.temperature
            
            # 🔥 Repetition Penalty Logic
            # هغه ټوکنونه چې لا دمخه کارول شوي، چانس یې کموي
            for token_id in set(generated[0].tolist()):
                if token_id == tokenizer.token_to_id(" د"):
                    logits[0, token_id] /= (req.repetition_penalty + 1.0) # د "د" لپاره اضافه جریمه
                else:
                    logits[0, token_id] /= req.repetition_penalty

            # Top-K Sampling
            top_k_logits, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
            logits[logits < top_k_logits[:, [-1]]] = -float('Inf')
            
            # Top-P (Nucleus) Sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > req.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = -float('Inf')
            
            # د احتمالاتو څخه انتخاب
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat((generated, next_token), dim=1)
            
            # که د جملې پای (End of Sentence) راغی
            if next_token.item() == tokenizer.token_to_id("</s>"):
                break
                
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

# ۵. د API برخې (Endpoints)
@app.get("/")
def home():
    return {"message": "Ghanam-1B Backend is running!", "status": "online"}

@app.post("/chat")
async def chat(request: GhanamRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="پرامپټ خالي دی!")
    
    try:
        raw_output = generate_response(request)
        # د کارېج ریټرن (\r) او اضافي سپېسونو پاکول
        clean_output = raw_output.replace('\r', '').replace('\n', ' ').strip()
        
        return {
            "prompt": request.prompt,
            "response": clean_output,
            "model": "Ghanam-1B-v0.1"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ۶. د سرور چالانول
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)