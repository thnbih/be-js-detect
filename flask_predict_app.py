# flask_predict_app.py
from flask import Flask, request, render_template_string
import esprima
import json
import torch
import torch.nn as nn
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# ---- Config ----
MODEL_PATH = "model_ast_structure_original.pt"
VOCAB_PATH = "vocab_ast_structure_original.json"
MAX_SEQ_LEN = 1000
EMBED_DIM = 100
NUM_CLASSES = 2

# ---- Load Vocab and Model ----
with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes)*num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conved = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))

model = TextCNN(len(vocab), EMBED_DIM, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---- Helper Functions ----
def extract_structure_sequence(js_code):
    try:
        tree = esprima.parseScript(js_code, tolerant=True)
    except Exception:
        return []
    sequence = []
    def dfs(node):
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type:
                sequence.append(node_type)
            for child in node.values():
                dfs(child)
        elif isinstance(node, list):
            for item in node:
                dfs(item)
    dfs(tree.toDict())
    return sequence

def encode_sequence(seq):
    if len(seq) > MAX_SEQ_LEN:
        seq = seq[:MAX_SEQ_LEN]
    else:
        seq += ['<pad>'] * (MAX_SEQ_LEN - len(seq))
    return [vocab.get(tok, vocab.get('<unk>', 1)) for tok in seq]

def predict_label(js_code):
    seq = extract_structure_sequence(js_code)
    if not seq:
        return "Could not parse JavaScript.", None
    indices = encode_sequence(seq)
    input_tensor = torch.tensor([indices], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        malicious_prob = probs[0, 1].item()  # Xác suất là malicious
    label = "malicious" if malicious_prob >= 0.5 else "benign"
    return label, malicious_prob

def crawl_first_js_from_url(url):
    try:
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script')
        for script in scripts:
            src = script.get('src')
            if src and src.endswith('.js'):
                js_url = src if src.startswith('http') else urljoin(url, src)
                js_response = requests.get(js_url, headers=headers, timeout=10, verify=False)
                if js_response.status_code == 200:
                    return js_response.text
                else:
                    return None
        return None
    except Exception:
        return None

# ---- Flask App ----
app = Flask(__name__)

HTML_PAGE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>JS Malware Detector</title>
  <style>
    body {
      background: #f7f7f7;
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
    }
    .container {
      max-width: 420px;
      margin: 40px auto;
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      padding: 32px 28px 24px 28px;
    }
    h2 {
      color: #333;
      margin-top: 0;
      margin-bottom: 18px;
      font-size: 1.2rem;
      font-weight: 600;
    }
    form {
      margin-bottom: 24px;
    }
    input[type="file"], input[type="text"] {
      width: 100%;
      padding: 8px;
      margin-bottom: 12px;
      border: 1px solid #ccc;
      border-radius: 5px;
      font-size: 1rem;
    }
    input[type="submit"] {
      background: #007bff;
      color: #fff;
      border: none;
      padding: 10px 18px;
      border-radius: 5px;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    input[type="submit"]:hover {
      background: #0056b3;
    }
    h3 {
      color: #007bff;
      margin-top: 18px;
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Upload JavaScript File</h2>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".js">
      <input type="submit" value="Predict">
    </form>

    <h2>Nhập url cần kiểm tra (có thể không lấy được file js)</h2>
    <form method="post">
      <input type="text" name="url" size="50" placeholder="https://example.com">
      <input type="submit" value="Scan">
    </form>

    {% if result %}
      <h3>Prediction: {{ result }}</h3>
    {% endif %}
  </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    probability = None
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file and file.filename.endswith('.js'):
                js_code = file.read().decode('utf-8')
                label, prob = predict_label(js_code)
                if prob is not None:
                    result = f"Prediction: {label} ({prob*100:.2f}%)"
                else:
                    result = label
            else:
                result = "Please upload a .js file."
        elif 'url' in request.form and request.form['url']:
            js_code = crawl_first_js_from_url(request.form['url'])
            if js_code:
                label, prob = predict_label(js_code)
                if prob is not None:
                    result = f"Prediction: {label} ({prob*100:.2f}%)"
                else:
                    result = label
            else:
                result = "Could not retrieve JavaScript from the provided URL."
    return render_template_string(HTML_PAGE, result=result)

if __name__ == '__main__':
    import urllib3
    urllib3.disable_warnings()

    import os
    port = int(os.environ.get("PORT", 5000))
