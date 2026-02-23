import ast
import csv
from openpyxl import Workbook
from pathlib import Path
import os

import logging
logger = logging.getLogger("CSV")
logger.setLevel(level=logging.INFO)


def convert_to_csv(input_file, output_file):
    rows = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = ast.literal_eval(line)

                rows.append(data)

    # Write to CSV
    fieldnames = ['TIN', 'Total', 'Date Issued', 'Company & Address', 'Link', 'Image Path']

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            # Filter only keys present in fieldnames
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)

    convert_to_xlsx(rows, "output/final_report.xlsx")

    logger.info(f"Converted to csv successfully! {output_file}")



def convert_to_xlsx(rows, output_file):
    wb = Workbook()
    ws = wb.active

    headers = [
        'TIN', 'Total', 'Date Issued',
        'Company & Address', 'Link',
        'Image Path'
    ]

    ws.append(headers)

    for row in rows:
        ws.append([row.get(h, "") for h in headers])
        current_row = ws.max_row

        img_col = headers.index("Image Path") + 1
        image_path_str = row.get("Image Path", "")

        if image_path_str:
            # Force correct relative path
            filename = Path(image_path_str).name
            correct_path = f"../receipts_v2/{filename}"

            cell = ws.cell(row=current_row, column=img_col)
            cell.value = "Open Receipt"
            cell.hyperlink = correct_path
            cell.style = "Hyperlink"

    wb.save(output_file)