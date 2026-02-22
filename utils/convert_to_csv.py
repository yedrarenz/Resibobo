import ast
import csv

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
    fieldnames = ['TIN', 'Total', 'Date Issued', 'Company & Address', 'Link']

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            # Filter only keys present in fieldnames
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)

    logger.info(f"Converted to csv successfully! {output_file}")