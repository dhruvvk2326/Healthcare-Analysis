import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import PyPDF2
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure nltk resources are downloaded
nltk.download('punkt')

# Function to read PDF
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    return text

# Expanded known diseases list (example)
known_diseases = [
    "COVID-19", "Influenza", "Ebola", "Malaria", "Tuberculosis", "Diabetes", 
    "Cancer", "HIV/AIDS", "Cholera", "Dengue", "Zika", "SARS", "MERS", 
    "Hypertension", "Asthma", "Alzheimer's Disease", "Parkinson's Disease",
    "Stroke", "Cardiovascular Disease", "Kidney Disease"
]

# Example expanded correlation matrix (20x20)
correlation_matrix = np.array([
    [1.0, 0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.4, 0.3, 0.2, 0.3, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 1.0, 0.3, 0.2, 0.3, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2, 0.3, 0.2, 0.1, 0.4, 0.5, 0.3, 0.2, 0.3],
    [0.2, 0.3, 1.0, 0.5, 0.4, 0.2, 0.3, 0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.4],
    [0.3, 0.2, 0.5, 1.0, 0.3, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.2, 0.3, 0.5, 0.4, 0.3, 0.2, 0.3],
    [0.4, 0.3, 0.4, 0.3, 1.0, 0.5, 0.4, 0.3, 0.2, 0.4, 0.5, 0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3],
    [0.5, 0.4, 0.2, 0.4, 0.5, 1.0, 0.5, 0.3, 0.2, 0.4, 0.3, 0.5, 0.4, 0.3, 0.2, 0.4, 0.5, 0.3, 0.4, 0.5],
    [0.6, 0.3, 0.3, 0.3, 0.4, 0.5, 1.0, 0.4, 0.3, 0.2, 0.4, 0.5, 0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 0.5, 0.4],
    [0.4, 0.2, 0.4, 0.5, 0.3, 0.3, 0.4, 1.0, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4],
    [0.3, 0.5, 0.3, 0.4, 0.2, 0.2, 0.3, 0.5, 1.0, 0.4, 0.3, 0.4, 0.5, 0.3, 0.4, 0.3, 0.2, 0.3, 0.4, 0.3],
    [0.2, 0.4, 0.2, 0.3, 0.4, 0.4, 0.2, 0.4, 0.4, 1.0, 0.3, 0.2, 0.3, 0.5, 0.4, 0.3, 0.2, 0.3, 0.2, 0.3],
    [0.3, 0.3, 0.4, 0.5, 0.5, 0.3, 0.4, 0.3, 0.3, 0.3, 1.0, 0.4, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.3, 0.2],
    [0.5, 0.2, 0.3, 0.4, 0.4, 0.5, 0.5, 0.2, 0.4, 0.2, 0.4, 1.0, 0.5, 0.3, 0.2, 0.5, 0.4, 0.3, 0.4, 0.5],
    [0.4, 0.3, 0.2, 0.3, 0.3, 0.4, 0.4, 0.3, 0.5, 0.3, 0.3, 0.5, 1.0, 0.4, 0.3, 0.2, 0.3, 0.5, 0.4, 0.3],
    [0.3, 0.2, 0.5, 0.2, 0.2, 0.3, 0.3, 0.4, 0.3, 0.5, 0.2, 0.3, 0.4, 1.0, 0.5, 0.4, 0.3, 0.2, 0.3, 0.2],
    [0.2, 0.1, 0.4, 0.3, 0.4, 0.2, 0.2, 0.5, 0.4, 0.4, 0.3, 0.2, 0.3, 0.5, 1.0, 0.3, 0.2, 0.3, 0.4, 0.5],
    [0.1, 0.4, 0.3, 0.5, 0.3, 0.4, 0.4, 0.4, 0.3, 0.3, 0.4, 0.5, 0.2, 0.4, 0.3, 1.0, 0.5, 0.4, 0.3, 0.2],
    [0.2, 0.5, 0.2, 0.4, 0.2, 0.5, 0.3, 0.3, 0.2, 0.2, 0.3, 0.4, 0.3, 0.3, 0.2, 0.5, 1.0, 0.3, 0.4, 0.3],
    [0.3, 0.3, 0.1, 0.3, 0.5, 0.3, 0.2, 0.2, 0.3, 0.3, 0.4, 0.3, 0.5, 0.2, 0.3, 0.4, 0.3, 1.0, 0.5, 0.4],
    [0.4, 0.2, 0.2, 0.2, 0.4, 0.4, 0.5, 0.3, 0.4, 0.2, 0.3, 0.4, 0.4, 0.3, 0.4, 0.3, 0.4, 0.5, 1.0, 0.5],
    [0.5, 0.3, 0.4, 0.3, 0.3, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.2, 0.3, 0.4, 0.5, 1.0]
])

# Function to extract diseases from text
def extract_diseases(text, known_diseases):
    words = word_tokenize(text)
    extracted_diseases = [disease for disease in known_diseases if disease.lower() in (word.lower() for word in words)]
    return extracted_diseases

# Function to mark diseases in the correlation matrix
def mark_diseases(extracted_diseases, known_diseases, correlation_matrix):
    marked_matrix = np.zeros_like(correlation_matrix)
    indices = [known_diseases.index(disease) for disease in extracted_diseases]
    marked_matrix[np.ix_(indices, indices)] = correlation_matrix[np.ix_(indices, indices)]
    return marked_matrix, indices

# Function to visualize correlation matrix
def plot_correlation_matrix(correlation_matrix, indices, known_diseases, file_name):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=known_diseases, yticklabels=known_diseases)
    for i in indices:
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=3))
    plt.title('Disease Correlation Matrix')
    plt.savefig(file_name)
    plt.close()

# Main function to process PDF and perform correlation analysis
def main(file_path):
    text = read_pdf(file_path)
    cleaned_text = clean_text(text)
    extracted_diseases = extract_diseases(cleaned_text, known_diseases)
    
    marked_matrix, indices = mark_diseases(extracted_diseases, known_diseases, correlation_matrix)
    
    file_name = 'correlation_matrix.png'
    plot_correlation_matrix(marked_matrix, indices, known_diseases, file_name)
    print(f"Correlation matrix saved as {file_name}")

# Example usage
file_path = "C:\\Users\\dell\\OneDrive\\Desktop\\NLP project\\Covid_research_paper.pdf"
main(file_path)
