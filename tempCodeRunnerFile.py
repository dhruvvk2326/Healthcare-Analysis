import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text
import PyPDF2
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download NLTK resources if not already downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to clean and preprocess text
def clean_text(text):
    # Remove links and emails
    text = re.sub(r'http\S+|www\S+|\S*@\S*\s?', '', text)

    # Normalize text and convert to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Stem the lemmatized tokens
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]

    # Remove stopwords and single-character tokens, but keep '%' sign
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in stemmed_tokens if word not in stop_words and (len(word) > 1 or word == '%')]

    # Join tokens back into text
    cleaned_text = ' '.join(filtered_tokens)

    return cleaned_text

# Specify the path to the PDF file
pdf_path = "Covid_research_paper.pdf"

# Specify the directory to save the cleaned text file
output_directory = os.path.dirname(pdf_path)

# Open the PDF file
with open(pdf_path, 'rb') as file:
    # Create a PDF reader object
    reader = PyPDF2.PdfReader(file)

    # Extract text from each page
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    # Clean and preprocess the extracted text
    cleaned_text = clean_text(text)

    # Specify the path to save the cleaned text file
    text_file_name = os.path.basename(pdf_path).replace('.pdf', '_cleaned.txt')
    text_file_path = os.path.join(output_directory, text_file_name)

    # Save the cleaned text as a text file
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(cleaned_text)

    print("Cleaned text saved to:", text_file_path)