# 📰 Fake News Detection with BERT + GPT Reasoning

This project is an advanced fake news classification system that combines the power of **BERT** (for prediction) with **Gemma/GPT-style reasoning** to explain whether a news article is real or fake — just like Grok AI!

---

## 🚀 Features

- ✅ Accurate prediction using fine-tuned BERT on `True.csv` and `Fake.csv`
- 🧠 LLM-based explanation using Google Gemma 3B/12B (via Ollama)
- 📊 Confidence scores for predictions
- ⚡ Fast Flask web interface with modern UI
- 📂 Organized directory structure for deployment & reproducibility

---

## 🏗️ Project Structure

FakeNews-BERT/
├── data/
│ ├── True.csv
│ └── Fake.csv
├── model/
│ └── best_model/ (BERT model files)
├── web_app/
│ ├── templates/
│ │ └── index.html
│ ├── static/
│ │ └── style.css
│ ├── app.py
│ └── llm_reasoning.py
├── bert_fakenews.py
├── predict.py
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 💻 How to Run

1. **Clone the repo**  
   ```bash
   git clone https://github.com/priyanshu044-4/FakeNews-BERT.git
   cd FakeNews-BERT
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Start Flask app

bash
Copy
Edit
cd web_app
python app.py
(Optional) Run Ollama for LLM reasoning

bash
Copy
Edit
ollama run gemma:12b
🌐 Web App Preview
Paste your news article, hit "Analyze" and see:

📢 Real or Fake prediction

💡 LLM reasoning behind it (Grok-style)

🤖 Model Training (BERT)
We use bert-base-uncased fine-tuned on the combined dataset of true and fake news articles:

512-token truncation

Adam optimizer

Epochs: 3–5

Custom bert_fakenews.py for training pipeline

📦 Requirements
Python 3.10+

Transformers

Torch

Flask

Ollama (for LLM reasoning)

Install all dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
🧠 LLM Reasoning (Gemma via Ollama)
To enable explanation with LLM:

Install Ollama

Pull the model:

bash
Copy
Edit
ollama pull gemma:12b
Ensure it's running on http://localhost:11434

If LLM response fails or times out, fallback is a default message.

📊 Dataset
Used only:

data/Fake.csv

data/True.csv

You don’t need to scrape or use external news APIs. This project is purely local.

👨‍💻 Author
Priyanshu Kumar
GitHub | LinkedIn

📄 License
MIT License
