# This project uses
- https://opencv.org/cropping-an-image-using-opencv/
- https://github.com/PaddlePaddle/PaddleOCR?tab=readme-ov-file
- https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec
- https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det

## Notes
- The project is expected for misspelling.
- The extraction depends on the image quality of the receipt.
- The parsing uses regular expression
  - There might be some issues with extraction if:
    - The receipt format is not yet added on the regular expression
    - Poor image quality.
- This is an open project, feel free to contribute.
- This uses PaddleOCR CPU, This is not yet optimized so running this can exhaust your CPU resources. :) 

## How to run
1. Add your receipts images that's already cropped (for easier extraction) to /receipts directory.
2. Install requirements
   - pip install -r requirements.txt
3. Run main.py
4. A CSV file will be generated on output directory.

## Sample Output:
| TIN               | Total   | Date Issued | Company & Address                                                                 | Link |
|-------------------|---------|------------|----------------------------------------------------------------------------------|------|
| 008-022-153-000   | 611.14  | 2026-02-16 | Grabit Foods Inc. Jollibee Waltermart Makiling Store #959 Brgy Makiling Nat Highway Calamba City | https://todo-sharepoint.com/1556d548-b279-443f-b392-376fcf23e15d.jpg |
| 009-433-354-000   | 1141.00 | 2619-02-16 | Caltex Sierra Makiling Gas Corporation Maharlika, Highway, San Antonio, Santo Tomas, Batangas | https://todo-sharepoint.com/517de74f-4b20-48d4-8734-1c122a1d5491.jpg |
| 000-122-954-000   | 2470.00 | 2026-02-04 | Greenfield Develooment Corp. Geenfield Tower Reenfield District, William St Cor. Mayflower St. Brgy. Highmay Hills Mandaluyong City | https://todo-sharepoint.com/61e6be3c-676c-4ff2-b7eb-27f9c4c1b1ee.jpg |
| 128-742-767-001   | 314.00  | 2026-02-13 | Tapa Kjng Operateo By:Keiser Foou Service | https://todo-sharepoint.com/7e428edc-0258-4efa-866f-5d0d14bd65a7.jpg |
| 009-433-354-000   | 1141.00 | 2619-02-16 | Caltex Sierra Makiling Gas Corporation Maharlika, Highway, San Atonio, Santo Tomas, Batangas | https://todo-sharepoint.com/b3fe9ccc-7f01-4d46-bfb5-d5b11d96874e.jpg |
| 128-742-767-001   | 239.00  | 2026-02-13 | Tapa King Operated By: Keiser Food Service. | https://todo-sharepoint.com/d2e0ee02-b168-4b4c-a489-b47d9398ae5a.jpg |
| 128-742-767-001   | 239.00  | 2026-02-13 | Tapa King Gperated By:Keiser Food Service. | https://todo-sharepoint.com/fc005eaa-bb51-40c0-ad9a-24e76506e1d6.jpg |


# Limitations
- Only works well with clear receipts.
- Image inferencing uses CPU for Windows 
- M1 / Mac Machines runs slow as it uses memory
- GPU is a must-have to run the script fast.

# For Improvements
- Data sanitization