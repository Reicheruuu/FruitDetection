import sys
import cv2
import imutils
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.uic import loadUiType
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize

# Load the UI file and extract the Ui_MainWindow class
ui, _ = loadUiType('UI.ui')

# Load the fruit detection model
fruits = torch.hub.load('ultralytics/yolov5', 'custom', path='Fruits/weights/best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fruits = fruits.to(device)
fruits.eval()

# Define the list of classes for detection
classes = ['bad apple', 'bad banana', 'bad orange', 'good apple', 'good banana', 'good orange']

# Define the confidence threshold for classification
confidence_threshold = 0.7  # Minimum confidence score to consider a detection

# Define known size ranges for bananas in inches
banana_small_range = (1, 3)
banana_medium_range = (4, 6)
banana_large_range = (7, 9)

# Define known size ranges for apples in inches
apple_small_range = (1, 2)
apple_medium_range = (2, 3)
apple_large_range = (4, 5)

# Define known size ranges for oranges in inches
orange_small_range = (2, 3)
orange_large_range = (4, 5)
# Define the MainApp class, which extends QMainWindow and Ui_MainWindow

class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        # Set the window title
        self.setWindowTitle("Fruit Detection")

        # Set the initial tab to the Tab 0
        self.tabWidget.setCurrentIndex(0)

        # Connect the login button to the login method
        self.MAIN_LOGIN.clicked.connect(self.login)

        # Connect the close button to the close_window method
        self.MAIN_CLOSE.clicked.connect(self.close_window)

        # Connect the detect button to the detect method
        self.DETECT.clicked.connect(self.detect)

        # Connect the stop button to the STOP method
        self.STOP.clicked.connect(self.stop)

        # Initialize the video capture object and the face cascade classifier
        self.cap = cv2.VideoCapture(1)

### LOGIN PROCESS ###
    def login(self):
        # Get the entered username and password
        un = self.USERNAME.text()
        un = un.lower()
        pw = self.PASSWORD.text()
        pw = pw.lower()

        # Check if the username and password are correct
        if (un == "erovoutika") and (pw == "123!"):
            # If the username and password are correct, clear the fields and switch to the main tab
            self.USERNAME.setText("")
            self.PASSWORD.setText("")
            self.tabWidget.setCurrentIndex(1)

        else:
            # If the username and/or password are incorrect, show an error message
            if (un != "erovoutika") and (pw == "123!"):
                msg = QMessageBox()
                msg.setText("The username you’ve entered is incorrect.")
                msg.setWindowTitle("INCORRECT USERNAME!")
                msg.setIcon(QMessageBox.Information)
                msg.setStyleSheet("background-color: rgb(36, 13, 74);color: rgb(255, 255, 255);")
                msg.exec_()
            elif (un == "erovoutika") and (pw != "123!"):
                msg = QMessageBox()
                msg.setText("The password you’ve entered is incorrect.")
                msg.setWindowTitle("INCORRECT PASSWORD!")
                msg.setIcon(QMessageBox.Information)
                msg.setStyleSheet("background-color: rgb(36, 13, 74);color: rgb(255, 255, 255);")
                msg.exec_()
            else:
                msg = QMessageBox()
                msg.setText("Please enter the correct username and password.")
                msg.setWindowTitle("USERNAME AND PASSWORD!")
                msg.setIcon(QMessageBox.Information)
                msg.setStyleSheet("background-color: rgb(36, 13, 74);color: rgb(255, 255, 255);")
                msg.exec_()

        # Clear the username and password fields
        self.USERNAME.setText("")
        self.PASSWORD.setText("")

    ### CLOSE WINDOW PROCESS ###
    def close_window(self):
        # Close the window
        self.close()

    ### DETECT BUTTON ###
    def detect(self):
        # Set the current tab to the Tab 1
        self.tabWidget.setCurrentIndex(1)
        self.fruit_detection()

    def fruit_detection(self):
        self.processing_video = True

        while self.processing_video:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Perform inference with fruit detection model
            results_fruits = fruits(frame)

            # Get detected objects, their positions, and confidence scores for fruit detection
            boxes = results_fruits.xyxy[0].cpu().numpy()
            labels = boxes[:, -1].astype(int)
            confidences = results_fruits.xyxy[0][:, 4].cpu().numpy()
            print(results_fruits)
            print(confidences)
            for i in range(len(boxes)):
                box = boxes[i]
                label = classes[labels[i]]
                confidence = confidences[i]

                # Filter detections based on confidence threshold
                if confidence > confidence_threshold:
                    x1, y1, x2, y2 = box[:4].astype(int)

                    # Calculate the size of the fruit in inches
                    width_pixels = x2 - x1
                    width_inches = (width_pixels / frame.shape[1]) * 9  # Assuming a max width of 9 inches

                    # Determine the size category based on fruit type
                    size_category = None
                    if 'banana' in label:
                        if banana_small_range[0] <= width_inches < banana_small_range[1]:
                            size_category = 'Small'
                        elif banana_medium_range[0] <= width_inches < banana_medium_range[1]:
                            size_category = 'Medium'
                        elif banana_large_range[0] <= width_inches < banana_large_range[1]:
                            size_category = 'Large'
                    elif 'apple' in label:
                        if apple_small_range[0] <= width_inches < apple_small_range[1]:
                            size_category = 'Small'
                        elif apple_medium_range[0] <= width_inches < apple_medium_range[1]:
                            size_category = 'Medium'
                        elif apple_large_range[0] <= width_inches < apple_large_range[1]:
                            size_category = 'Large'
                    elif 'orange' in label:
                        if orange_small_range[0] <= width_inches < orange_small_range[1]:
                            size_category = 'Small'
                        elif orange_large_range[0] <= width_inches < orange_large_range[1]:
                            size_category = 'Large'

                    # Define bounding box color based on fruit quality
                    if 'good' in label:
                        box_color = (0, 255, 0)  # Green for good fruits
                    else:
                        box_color = (255, 0, 0)  # Red for bad fruits

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f'{label} ({width_inches:.2f} inches, {size_category})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    # Update the GUI to display the recognized fruit's information
                    self.FRUIT_NAME.setText(label)
                    self.FRUIT_SIZE.setText(str(width_inches))  # Convert to string
                    self.FRUIT_CATEGORY.setText(size_category)

            # Convert the frame to BGR color space
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Convert the frame to QImage
            qImg = QImage(frame_bgr.data, frame_bgr.shape[1], frame_bgr.shape[0], frame_bgr.strides[0],
                          QImage.Format_RGB888)

            # Create a QPixmap from QImage
            pixmap = QPixmap.fromImage(qImg)

            # Set the pixmap to be scaled to the size of the QLabel widget
            self.FRUIT_CAPTURE.setScaledContents(True)

            # Set the pixmap as the image displayed in the QLabel widget
            self.FRUIT_CAPTURE.setPixmap(pixmap)

            # Repaint the QLabel widget
            self.FRUIT_CAPTURE.repaint()

            # Process any pending events in the application's event queue
            QApplication.processEvents()

        self.cap.release()

    def stop(self):
        self.processing_video = False
        self.cap.release()

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


def main():
    # Create an instance of the QApplication class
    app = QApplication(sys.argv)

    # Create an instance of the MainApp class (the main window of the program)
    window = MainApp()

    # Show the window
    window.show()
    # Start the event loop of the application
    app.exec_()

if __name__ == '__main__':
    # Call the main function if this script is being run as the main program
    main()