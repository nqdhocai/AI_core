import docx
import pandas as pd
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_docx(file_path):
    """Trích xuất văn bản từ file DOCX."""
    doc = docx.Document(file_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text)

def extract_text_from_txt(file_path):
    """Trích xuất văn bản từ file TXT."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_excel(file_path, sheet_name=0):
    """Trích xuất văn bản từ file Excel."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df.to_string()

def extract_text_from_pdf(file_path):
    """Trích xuất văn bản từ file PDF."""
    text = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text.append(page.extract_text())
    return '\n'.join(text)

def extract_text(file_path):
    file_type = file_path.split('.')[-1]
    case = {
        "txt": extract_text_from_txt(file_path),
        "docx": extract_text_from_docx(file_path),
        "pdf": extract_text_from_pdf(file_path),
        "excel": extract_text_from_excel(file_path),
    }
    return case[file_type]

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Phân chia văn bản thành các chunk nhỏ hơn.

    :param text: Văn bản cần phân chia
    :param chunk_size: Kích thước tối đa của mỗi chunk
    :param chunk_overlap: Kích thước chồng lấp giữa các chunk
    :return: Danh sách các chunk văn bản
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split(text)