import PyPDF2

def pdf_to_text(pdf_path, txt_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text_lines = []

        # Iterate through each page
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            text_lines.extend(text.splitlines())

    # Write the text to a TXT file
    with open(txt_path, 'w') as txt_file:
        for line in text_lines:
            txt_file.write(line + '\n')

if __name__ == "__main__":
    pdf_path = 'textbook.pdf'
    txt_path = 'output.txt'
    pdf_to_text(pdf_path, txt_path)