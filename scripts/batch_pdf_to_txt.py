import os
from pathlib import Path
import PyPDF2
import json

# OCR dependencies
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_txt(pdf_path, txt_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                else:
                    # OCR fallback for pages without extractable text
                    try:
                        # pdf2image returns a list (even if single page)
                        images = convert_from_path(str(pdf_path), first_page=i+1, last_page=i+1)
                        if images:
                            ocr_text = pytesseract.image_to_string(images[0])
                            if ocr_text.strip():
                                text += "[[OCR]] " + ocr_text + "\n"
                    except Exception as ocr_e:
                        print(f"[OCR ERROR] Page {i+1} of {pdf_path}: {ocr_e}")
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
        print(f"[PDF → TXT] {pdf_path} → {txt_path}")
    except Exception as e:
        print(f"[ERROR] Failed to convert {pdf_path}: {e}")

def batch_convert_pdfs(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".pdf"):
                pdf_path = Path(dirpath) / file
                txt_path = pdf_path.with_suffix(".txt")
                pdf_to_txt(pdf_path, txt_path)

def summarize_json_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".json"):
                json_path = Path(dirpath) / file
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        print(f"[JSON] {json_path}: {len(data)} keys")
                    elif isinstance(data, list):
                        print(f"[JSON] {json_path}: {len(data)} items (list)")
                    else:
                        print(f"[JSON] {json_path}: EMPTY or unrecognized structure")
                except Exception as e:
                    print(f"[ERROR] Failed to read {json_path}: {e}")

if __name__ == "__main__":
    for subdir in ["data/training", "data/validation"]:
        base_dir = Path(subdir)
        print(f"\n--- Converting PDFs to TXT in {subdir} ---\n")
        batch_convert_pdfs(base_dir)
        print(f"\n--- Summarizing JSON files in {subdir} ---\n")
        summarize_json_files(base_dir)
