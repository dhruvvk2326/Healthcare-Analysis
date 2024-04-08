import pypdf as pdf
import spacy

file = open('Healthcare Study Comparison and Analysis Tool Proposal.pdf','rb')
pdf_reader= pdf.PdfReader(file)

page_content={}

for index, pdf_page in enumerate(pdf_reader.pages):
    page_content[index+1] =  pdf_page.extract_text()
    
print(page_content)

y= pdf_reader.is_encrypted
print(y)

nlp = spacy.load("en_core_web_sm")

def extract_words(text):
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha] 
    return words

all_words = []
for index, page_text in page_content.items():
    words_on_page = extract_words(page_text)
    all_words.extend(words_on_page)

print(all_words)

#Biobert usage from here

from transformers import AutoTokenizer, AutoModel
import torch

# Load BioBERT model and tokenizer
model_name = "monologg/biobert_v1.1_pubmed"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to preprocess text for BioBERT
def preprocess_text(text):
    # Tokenize the text
    tokens = text.split()
    # Lowercase tokens
    tokens = [token.lower() for token in tokens]
    # Remove non-biomedical terms (you may need to define a list of biomedical terms)
    # For simplicity, we'll skip this step in this example
    return " ".join(tokens)

# Function to extract words from text using spaCy
def extract_words(text):
    # Process the text with spaCy
    doc = nlp(text)
    # Extract tokens (words) from the document
    words = [token.text for token in doc if token.is_alpha]  # Filter out non-alphabetic tokens
    return words

# Extract words from each page of the PDF and preprocess for BioBERT
preprocessed_texts = []
for index, page_text in page_content.items():
    words_on_page = extract_words(page_text)
    text = " ".join(words_on_page)
    preprocessed_text = preprocess_text(text)
    preprocessed_texts.append(preprocessed_text)

# Tokenize and encode preprocessed text using BioBERT tokenizer
encoded_inputs = tokenizer(preprocessed_texts, padding=True, truncation=True, return_tensors="pt")

# Pass inputs through BioBERT model
with torch.no_grad():
    outputs = model(**encoded_inputs)

# Extract contextual embeddings (last hidden states)
embeddings = outputs.last_hidden_state

# Now you can analyze the embeddings and extract insights as needed
