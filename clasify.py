import pickle
import torch
from torch import nn
from nltk.tokenize import word_tokenize
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

print(BASE_DIR)

# ── 1. Load vocab and config ──────────────────────────────────────────────────
with open(BASE_DIR / 'vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open(BASE_DIR / 'config.pkl', 'rb') as f:
    config = pickle.load(f)

max_len = config['max_len']


# ── 2. Copy the exact same model class ───────────────────────────────────────
class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout    = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc         = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        pooled = lstm_out.mean(dim=1)
        pooled = self.dropout(pooled)
        pooled = self.layer_norm(pooled)
        return self.fc(pooled)
    

    # ── 3. Load the saved weights ─────────────────────────────────────────────────
model = LSTMmodel(
    vocab_size = config['vocab_size'],
    embed_dim  = 100,
    hidden_dim = 256,
    output_dim = 6
)
model.load_state_dict(torch.load(BASE_DIR / 'best_model.pt', map_location='cpu'))
model.eval()
print("Model loaded successfully")



# ── 4. Copy the same preprocessing functions ──────────────────────────────────
from nltk.corpus import stopwords
KEEP_NEGATIONS = {"not","no","nor","never","don't","didn't","won't","can't","couldn't","shouldn't","wouldn't"}
stop_words = set(stopwords.words('english')) - KEEP_NEGATIONS

def clear_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words]

def encode(tokens):
    return [vocab.get(w, 1) for w in tokens]

def pad(seq):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))



# ── 5. Predict on a single sentence ──────────────────────────────────────────
label_map = {0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"}

def predict(text):
    tokens  = clear_text(text)
    encoded = encode(tokens)
    padded  = pad(encoded)
    tensor  = torch.tensor([padded], dtype=torch.long)   # shape [1, max_len]

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = probs.argmax().item()

    return {
        'emotion'     : label_map[pred],
        'confidence'  : f"{probs[pred]*100:.1f}%",
        'all_scores'  : {label_map[i]: f"{p*100:.1f}%" for i, p in enumerate(probs)}
    }



# ── 6. Test it ────────────────────────────────────────────────────────────────
print(predict("I am so happy today!"))
print(predict("I can't believe they did that to me"))
