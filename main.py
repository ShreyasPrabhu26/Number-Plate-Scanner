import cv2
from PIL import Image
import pytesseract
from generate_barcode import generate_barcode

# Path for Harcascade Model
harcascade = "model/haarcascade_russian_plate_number.xml"

# Path for Pytesseract Engine
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Initialize OpenCV window
capture = cv2.VideoCapture(0)
capture.set(3, 640)  # Set width
capture.set(4, 480)  # Set height

# Initialize minimum area
minArea = 500
# Initialize image counter for saving cropped number plate portion
imageNumber = 1

while True:
    success, frame = capture.read()

    # Check if the frame was read successfully
    if not success:
        print("Error: Could not read frame.")
        break

    # Load Cascade File
    plateCascade = cv2.CascadeClassifier(harcascade)

    # Convert Image to Grey
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get Plate Coordinates
    plateCoordinates = plateCascade.detectMultiScale(grayImage, 1.1, 4)

    for (x, y, w, h) in plateCoordinates:
        # Calculate the plate Area
        detectedArea = w * h

        if detectedArea > minArea:
            # Draw rectangle around the detected plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show "Number Plate" text on top of the plate
            cv2.putText(frame, "Number Plate", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Crop Coordinates of the Number Plate Portion
            NumberPlateFrameCoordinates = frame[y:y + h, x:x + w]
            cv2.imshow("NumberPlateFrame", NumberPlateFrameCoordinates)

    cv2.imshow("Number Plate Detector", frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Saving Croped Number Plate Portion for OCR
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Save the cropped image in the Local Folder
        cv2.imwrite("plateImages/scaned_img_" + str(imageNumber) + ".jpg", NumberPlateFrameCoordinates)

        # Display "Plate Saved" message on the frame
        cv2.rectangle(frame, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, "Plate Saved", (150, 265),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", frame)
        cv2.waitKey(500)

        # Open the saved image using Pillow (PIL)
        image_path = "plateImages/scaned_img_" + str(imageNumber) + ".jpg"
        img = Image.open(image_path)

        # Perform OCR on the image
        text = pytesseract.image_to_string(img)
        output_file_path = 'extracted_text.txt'

        # Write the extracted text to a text file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(text)

        print(text)
        print(f'Text saved to {output_file_path}')

        generate_barcode()

        imageNumber += 1

# Release the VideoCapture and close the OpenCV window
capture.release()
cv2.destroyAllWindows()
