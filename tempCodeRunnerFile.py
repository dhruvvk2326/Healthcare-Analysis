import os
import PyPDF2

# Specify the path to the PDF file
pdf_path = "Covid_research_paper.pdf"

# Specify the directory to save the extracted text file
text_directory = "C:\\Users\\dhruv\\OneDrive\\Desktop\\Project\Healthcare-Analysis"

# Open the PDF file
with open(pdf_path, 'rb') as file:
    # Create a PDF reader object
    reader = PyPDF2.PdfReader(file)

    # Extract text from each page
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    # Save the extracted text as a text file with UTF-8 encoding
    text_file_name = os.path.basename(pdf_path).replace('.pdf', '.txt')
    text_file_path = os.path.join(text_directory, text_file_name)
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

print("Text extracted and saved successfully!")
