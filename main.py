# https://opencv.org/cropping-an-image-using-opencv/
# https://github.com/PaddlePaddle/PaddleOCR?tab=readme-ov-file
# https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec
# https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det

import os

from pandas.conftest import cls

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 30).__str__()
os.environ['DISABLE_AUTO_LOGGING_CONFIG'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import re
from dateutil import parser

from PIL import Image

from paddleocr import PaddleOCR, logger
import logging
logger.setLevel(level=logging.INFO)

# For future feature
def crop_receipt(image_path):
    """
    This is just for cropping an image
    :param image_path:
    :return:
    """
    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]
    original_dim = (original_width, original_height)

    img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)

    # Let user select ROI (drag a box)
    roi = cv2.selectROI("Select ROI", img, False)

    # Extract cropped region
    cropped_img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    # Save and display cropped image
    # set_original_size = cv2.resize(cropped_img, original_dim, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(f"{image_path.split('.')[0]}-cropped.png", cropped_img)
    cv2.imshow("Cropped Image", cropped_img)
    cv2.destroyAllWindows()

# For future feature
def sharpen_image(image):
    import numpy as np


    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # 3. Remove noise
    blur = cv2.GaussianBlur(contrast, (3, 3), 0)

    # 4. Adaptive threshold (VERY important)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 5. Optional sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    return sharpened

def text_extractor(image_path):
    """
    Just extract text from an image
    :param image_path:
    :return:
    """

    # Run OCR inference on a sample image
    result = ocr.predict(
        input=image_path)

    # Visualize the results and save the JSON results
    # for res in result:
    #     #res.print()
    #     res.save_to_img("misc")
    #     res.save_to_json("misc")
    return result

def tin_formatter(full_text, data):
    # TIN patterns
    # To Do Branch code at last digits should be 5 = 00000
    tin_patterns = [r'\d{3}-\d{3}-\d{3}-\d{3}',
                    r'\bTIN[:.]?\s*([\d\-]+)',
                    r'VAT REG TIN.[:.]?\s*([\d\-]+)',
                    r'T1N[:.]?\s*([\d\-]+)']
    for pat in tin_patterns:
        m = re.search(pat, full_text, flags=re.IGNORECASE)
        if m:
            data['TIN'] = m.group(0)
            break

def total_formatter(full_text, data):
    # Total patterns
    total_patterns = [
        r'\bTotal\s*[\(\（].*?VAT.*?[\)\）]\s*(?:₱|Php|P)?\s*([\d,]+(?:\.\d{1,2})?)',
        r'\bTotal\s*Due\b[:.]?\s*(?:₱|Php|P)?\s*([\d,]+(?:\.\d{1,2})?)',
        r'\bTotal\s+Invoice\b[:.]?\s*(?:₱|Php|P)?\s*([\d,]+(?:\.\d{1,2})?)',
        r'\bAmount\s*\bDue\s*[:.](?:₱|Php|P)?\s*(?:₱|Php|P)?([\d,]+\.?\d*)',
        r'\bTotal\b[:.]?\s*(?:₱|Php|P)?\s*([\d,]+(?:\.\d{1,2})?)'
    ]

    for pat in total_patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            data['Total'] = m.group(1)
            break

def company_formatter(data, lines):
    company_lines = []

    for i, line in enumerate(lines):
        line = line.strip()
        # Look for TIN / VAT / MIN keywords
        if re.search(r'TIN|VAT|MIN|ACCRED', line, flags=re.IGNORECASE):
            # Try capturing **after TIN** as address if company not already captured
            address_lines = []
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line or re.search(r'SN|MIN|Invoice|Date|Time|Cash|Total', next_line, re.I):
                    break

                # Remove common unwanted prefixes
                next_line = re.sub(r'Address:|ADDRESS:|To#|Hin:|Min:|Shid|Pcs/N|SN|Accred|VAT REG TIN',
                                   '', next_line, flags=re.I)

                # Remove pattern like ":Pc03Zwjh-B-1" (colon + alphanumeric + optional dashes)
                next_line = re.sub(r':\s*[A-Za-z0-9\-]+', '', next_line)

                # Remove long digit sequences (8+ digits)
                next_line = re.sub(r'\b\d{8,}\b', '', next_line)

                # Remove extra punctuation from OCR artifacts
                next_line = re.sub(r'[_*|<>]', ' ', next_line)

                # Collapse multiple spaces
                next_line = re.sub(r'\s+', ' ', next_line).strip()

                if next_line:
                    address_lines.append(next_line.title())

            if address_lines:
                company_lines.extend(address_lines)
            break
        else:
            company_lines.append(line.title())

    if company_lines:
        data['Company & Address'] = " ".join(company_lines)

def date_formatter(full_text, data):
    # Date patterns
    date_patterns = [
        r'\b\d{4}[./-]\d{2}[./-]\d{2}',  # 2026-01-11 or 2026.01.11
        r'\b\d{2}/\d{2}/\d{2,4}',  # 01/11/2026
        r'\b[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}',  # January 11, 2026
        r'\b\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\b' # 10 Feb 2026
    ]

    candidates = []

    for pat in date_patterns:
        matches = re.findall(pat, full_text)
        if matches:
            candidates.append(matches[0])

        # Remove dates that are likely accreditation dates
        filtered = []
        for date in candidates:
            # check nearby words
            context_pattern = rf'.{{0,30}}{re.escape(date)}.{{0,30}}'
            context = re.search(context_pattern, full_text)
            if context:
                snippet = context.group(0)
                if re.search(r'Issued|Accred|Accreditation', snippet, flags=re.IGNORECASE):
                    continue
            filtered.append(date)
            data['Date Issued'] = normalize_date(filtered[0])
            break

def extract_biz_info(image_path):
    result = text_extractor(str(image_path))
    lines = result[0]['rec_texts']
    full_text = " ".join(lines)
    data = {}

    logger.info(f'Parsing extracted receipt: {full_text}')

    logger.info(f"Extracting from {image_path}...")
    logger.info("Extracting TIN...")
    tin_formatter(full_text, data)

    logger.info("Extracting Total...")
    total_formatter(full_text, data)

    logger.info("Extracting Date Issued...")
    date_formatter(full_text, data)

    logger.info("Extracting Company & Address...\n\n")
    company_formatter(data, lines)

    logger.info(f'Extracted data: {data}\n\n')

    return data

def normalize_date(date_str):
    try:
        date_str = date_str.strip()

        # Remove words like "INVOICE"
        date_str = re.sub(r'[A-Za-z]+', lambda x: x.group() if x.group().lower() in
                          ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
                          else '', date_str)

        # If Date has format '/' and 202600 -- Last 2 Digits came from time.
        if '/' in date_str:
            if len(date_str.split('/')[2]) == 6:
                date_str = date_str[:-2]

        # Parse automatically
        parsed = parser.parse(date_str)
        return parsed.strftime("%Y-%m-%d")

    except Exception:
        return None

if __name__ == "__main__":
    from pathlib import Path
    from utils.convert_to_excel import convert_to_excel

    ocr = PaddleOCR(
        text_detection_model_dir="models/PP-OCRv5_server_det",
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir="models/PP-OCRv5_server_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en'
    )

    dir_path = Path("receipts")

    final_report = ""

    for file_path in dir_path.iterdir():
        if file_path.is_file():
            info = extract_biz_info(file_path)
            info['Link'] = f"https://todo-sharepoint.com/{file_path.name}"
            final_report += f"{info}\n"

    with open("final_report/report.txt", "w") as file:
        file.write(final_report)

    convert_to_excel("final_report/report.txt", "output/final_report.csv")
