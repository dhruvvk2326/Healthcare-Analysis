import pypdf as pdf

file = open('Healthcare Study Comparison and Analysis Tool Proposal.pdf','rb')
pdf_reader= pdf.PdfReader(file)

page_content={}

for index, pdf_page in enumerate(pdf_reader.pages):
    page_content[index+1] =  pdf_page.extract_text()
    
print(page_content)

y= pdf_reader.is_encrypted
print(y)