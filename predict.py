import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import re

# 🔧 Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-z0-9\s.,!?']", "", text)  # remove special chars
    return text.lower().strip()

# 🎨 Colored terminal output
def colored(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

# 🧠 Load tokenizer and model
model_path = "model/best_model"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# ⚙️ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔍 Prediction function
def predict_news(news_text):
    news_text = clean_text(news_text)
    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label = "🟢 REAL News" if pred == 1 else "🔴 FAKE News"
    confidence = probs[0][pred].item()
    return label, round(confidence * 100, 2)

# 🧪 Test multiple times
print(colored("🔥 Fake News Detector (BERT)\n", "96"))
while True:
    news = input(colored("📝 Paste news/article (or type 'exit'): ", "93"))
    if news.strip().lower() == "exit":
        print(colored("👋 Exiting. Stay sharp!", "90"))
        break

    label, confidence = predict_news(news)
    
    color = "92" if "REAL" in label else "91"
    print(f"\n🔎 Prediction: {colored(label, color)}")
    print(f"📊 Confidence: {colored(str(confidence) + '%', '94')}")
