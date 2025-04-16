from flask import Flask, request, jsonify, render_template
from transformers import BertForTokenClassification, BertTokenizer, pipeline, AutoModelForTokenClassification, AutoTokenizer
import cv2
import pytesseract
import re
import os

app = Flask(__name__)

# Ensure 'uploads' directory exists
os.makedirs('uploads', exist_ok=True)

# Load the fine-tuned BERT model
bert_model_name = "bert-model"
try:
    bert_model = BertForTokenClassification.from_pretrained(bert_model_name)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_basic_tokenize=True)
    bert_pipeline = pipeline(task="ner", model=bert_model, tokenizer=bert_tokenizer, device=-1)  # Use CPU
except Exception as e:
    print(f"Error loading BERT model: {str(e)}")
    exit()


#hf_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
hf_model_name = "eventdata-utd/conflibert-named-entity-recognition"
try:
    hf_model = AutoModelForTokenClassification.from_pretrained(hf_model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_pipeline = pipeline(task="ner", model=hf_model, tokenizer=hf_tokenizer, device=-1)  # Use CPU
except Exception as e:
    print(f"Error loading Hugging Face model: {str(e)}")
    exit()

# Serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Image upload and text extraction
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        # Process image with OCR
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        # Format extracted text
        formatted_text = re.sub(r'\s+', ' ', text)
        formatted_text = re.sub(r'\n+', ' ', formatted_text)
        formatted_text = re.sub(r'[^a-zA-Z0-9.,!?;:\'\"()\s-]', '', formatted_text)
        formatted_text = formatted_text.strip().capitalize()

        os.remove(file_path)  # Clean up uploaded file

        return jsonify({"extracted_text": formatted_text})

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

# Entity extraction route
@app.route('/extract_entities', methods=['POST'])
def extract_entities():
    data = request.get_json()
    text = data.get('text', '')
    model_choice = data.get('model', 'bert')  # Default to BERT

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # Choose the appropriate model based on user selection
        if model_choice == 'bert':
            entities = bert_pipeline(text)
        elif model_choice == 'huggingface':
            entities = hf_pipeline(text)
        else:
            return jsonify({"error": "Invalid model choice."}), 400

        # Post-processing to merge tokens
        merged_entities = []
        current_word = ""
        current_entity = None

        for entity in entities:
            word = entity['word']
            label = entity['entity'].split("-")[-1]  # Extract entity name

            if word.startswith("##"):
                current_word += word[2:]
            else:
                if current_word and current_entity:
                    merged_entities.append({"word": current_word, "entity": current_entity})
                current_word = word
                current_entity = label

        if current_word and current_entity:
            merged_entities.append({"word": current_word, "entity": current_entity})

        # Group words by entity type
        grouped_entities = {}
        previous_label = None

        for item in merged_entities:
            word = item['word']
            label = item['entity']

            if label != previous_label:
                if label not in grouped_entities:
                    grouped_entities[label] = [word]
                else:
                    grouped_entities[label].append(word)
            else:
                grouped_entities[label][-1] += f" {word}"

            previous_label = label

        return jsonify({"entities": grouped_entities})

    except Exception as e:
        return jsonify({"error": f"Failed to extract entities: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
