# https://opencv.org/cropping-an-image-using-opencv/
# https://github.com/PaddlePaddle/PaddleOCR?tab=readme-ov-file

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 30).__str__()
os.environ['DISABLE_AUTO_LOGGING_CONFIG'] = '1'

import cv2
import numpy as np
import re
from dateutil import parser

from paddleocr import PaddleOCR, logger
import logging, sys
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
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

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
    total_patterns = [r'Total\s*Due[:.]?\s*P?\s*([\d,]+\.?\d*)',
                      r'Total\s*(?:Due|Amount)?[:.]?\s*P\s*([\d,]+\.?\d*)',
                      r'([\d,]+\.?\d*)\s*TOTAL\s*(?:AMOUNT\s*)?DUE',
                      r'Total[:.]?\s*P?([\d,.]+)']

    for pat in total_patterns:
        m = re.search(pat, full_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            total = re.findall(r'\d+(?:\.\d+)?', m.group(0))
            data['Total'] = total[0]
            break

def company_formatter(data, lines):
    # Company & Address: take first 2-3 lines before any TIN/VAT/MIN keywords
    company_lines = []
    for line in lines:
        if re.search(r'TIN|VAT|MIN|ACCRED', line, flags=re.IGNORECASE):
            break
        company_lines.append(line.strip().title())
    if company_lines:
        data['Company & Address'] = " ".join(company_lines)

def date_formatter(full_text, data):
    # Date patterns
    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}(?!\d)',
        r'\b\d{2}/\d{2}/\d{2,4}(?!\d)',
        r'\b[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}\b'
    ]

    candidates = []

    for pat in date_patterns:
        matches = re.findall(pat, full_text)
        candidates.extend(matches)

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

    logger.info(f"Extracting from {image_path}...")
    logger.info("Extracting TIN...")
    tin_formatter(full_text, data)
    logger.info("Extracting Total...")
    total_formatter(full_text, data)
    logger.info("Extracting Date Issued...")
    date_formatter(full_text, data)
    logger.info("Extracting Company & Address...")
    company_formatter(data, lines)


    return data

def normalize_date(date_str):
    try:
        date_str = date_str.strip()

        # Remove words like "INVOICE"
        date_str = re.sub(r'[A-Za-z]+', lambda x: x.group() if x.group().lower() in
                          ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
                          else '', date_str)

        # Parse automatically
        parsed = parser.parse(date_str)
        return parsed.strftime("%Y-%m-%d")

    except Exception:
        return None

if __name__ == "__main__":
    from pathlib import Path
    from utils.convert_to_excel import convert_to_excel

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
