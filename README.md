
# Named Entity Recognition

This project implements a machine learning-based Named Entity Recognition (NER) system that identifies and categorizes entities (like names, dates, organizations, locations, etc.) in a given text or image. It uses a fine-tuned BERT model and integrates OCR for text extraction from images.


## Run Locally
- Use python 3.10
  
Clone the project

```bash
  git clone https://github.com/raju2525/BERT_NAMED-ENTITY-RECOGNITION.git
```

Go to the project directory

```bash
  cd BERT_NAMED-ENTITY-RECOGNITION
```

Install dependencies

```bash
   pip install -r requirements.txt
```

Start the server

```bash
 python app.py
```


## Running Tests

To run tests, run the following command

```bash
  pyton app.py
```


## Screenshots


![App Screenshot](https://raw.githubusercontent.com/raju2525/BERT_NAMED-ENTITY-RECOGNITION/main/screenshots/ss1.png)

![App Screenshot](https://raw.githubusercontent.com/raju2525/BERT_NAMED-ENTITY-RECOGNITION/main/screenshots/ss2.png)

![App Screenshot](https://raw.githubusercontent.com/raju2525/BERT_NAMED-ENTITY-RECOGNITION/main/screenshots/ss3.png)

![App Screenshot](https://raw.githubusercontent.com/raju2525/BERT_NAMED-ENTITY-RECOGNITION/main/screenshots/ss4.png)


## Features

🔡 Text and image-based NER

📑 Extracts entities like PERSON, LOCATION, ORGANIZATION, DATE, etc.

🧠 BERT fine-tuned for NER tasks

📷 OCR (Optical Character Recognition) for image input




## Tech Stack
**Frontend:** React, Tailwind CSS

**Backend:** Flask, Python

**ML Model:** PreTrained (BERT),Other model  used pipeline from huggingface 

**OCR:** Tesseract

**Deployment:** AWS EC2 (t3.micro), Nginx, Gunicorn
## AWS Deployment
- We Deployed our project into AWS Cloud EC2
- For detailed step by step of deploying your model/project into AWS refer for the pdf file available in project folder
- 
## Author
**RAJU BANDAM**

**EMAIL :** rajubandam694@gmail.com


