"""from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

MODEL_PATH = "../model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

@app.route("/")
def home():
    return "Toxicity Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    comment = data["comment"]

    inputs = tokenizer(comment, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0].tolist()

    result = dict(zip(labels, probs))

    result["is_toxic"] = any(p > 0.5 for p in probs)

    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5000)
    





from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

MODEL_NAME = "Aakrithi19/comment-toxicity-detector"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

@app.route("/")
def home():
    return "Toxicity Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    comment = data["comment"]

    inputs = tokenizer(comment, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0].tolist()

    result = dict(zip(labels, probs))
    result["is_toxic"] = any(p > 0.5 for p in probs)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)






from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

MODEL_NAME = "Aakrithi19/comment-toxicity-detector"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

@app.route("/")
def home():
    return "Toxicity Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    comment = data["comment"]

    inputs = tokenizer(comment, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0].tolist()

    result = dict(zip(labels, probs))
    result["is_toxic"] = any(p > 0.5 for p in probs)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

  





from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

HF_API_URL = "https://api-inference.huggingface.co/models/Aakrithi19/comment-toxicity-detector"

import os

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}
labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


@app.route("/")
def home():
    return "Toxicity Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    comment = data["comment"]

    payload = {"inputs": comment}

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    result = response.json()

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

    """



from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/Aakrithi19/comment-toxicity-detector"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.route("/")
def home():
    return "Toxicity Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    comment = data.get("comment","")

    try:

        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": comment}
        )

        result = response.json()

        print("HF response:", result)

        scores = {
            "toxic":0,
            "insult":0,
            "obscene":0,
            "identity_hate":0
        }

        # Handle model loading case
        if isinstance(result, dict) and "error" in result:
            return jsonify(scores)

        # HuggingFace sometimes returns [[...]]
        if isinstance(result, list):

            if len(result) > 0 and isinstance(result[0], list):
                result = result[0]

            for item in result:

                label = item.get("label","").lower()
                score = item.get("score",0)

                if label == "toxic":
                    scores["toxic"] = score

                elif label == "insult":
                    scores["insult"] = score

                elif label == "obscene":
                    scores["obscene"] = score

                elif label == "identity_hate":
                    scores["identity_hate"] = score

        return jsonify(scores)

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "toxic":0,
            "insult":0,
            "obscene":0,
            "identity_hate":0
        })


# IMPORTANT: No app.run() because Render runs via gunicorn