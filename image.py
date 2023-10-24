import cv2
import numpy as np
import torch
from torchvision.transforms.functional import resize
from PIL import Image

# Load the fruit detection model
fruits = torch.hub.load('ultralytics/yolov5', 'custom', path='Fruits/weights/best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fruits = fruits.to(device)
fruits.eval()

# Define the list of classes for detection
classes = ['bad apple', 'bad banana', 'bad orange', 'good apple', 'good banana', 'good orange']

# Define the size ranges for banana, apple, and orange in inches
banana_small_range = (6, 7)
banana_medium_range = (7, 8)
banana_large_range = (8, 9)

# Define the diameter ranges for apples in inches
apple_small_diameter_range = (1, 2)
apple_medium_diameter_range = (2, 3)
apple_large_diameter_range = (4, 5)

orange_small_range = (2, 3)
orange_large_range = (4, 5)

# Define the confidence threshold for classification
confidence_threshold = 0.7  # Minimum confidence score to consider a detection

# Load the image
image_path = '3.jpg'  # Replace with the path to your image
frame = cv2.imread(image_path)

# Resize the frame
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
height, width, _ = frame.shape
max_size = 900
if width > height:
    new_width = max_size
    new_height = int(height / (width / max_size))
else:
    new_height = max_size
    new_width = int(width / (height / max_size))
frame = resize(Image.fromarray(frame), (new_height, new_width))

# Convert frame to numpy array
frame = np.array(frame)

# Perform inference with fruit detection model
results_fruits = fruits(frame)

# Get detected objects, their positions, and confidence scores for fruit detection
boxes = results_fruits.xyxy[0].cpu().numpy()
labels = boxes[:, -1].astype(int)
confidences = results_fruits.xyxy[0][:, 4].cpu().numpy()

# Initialize lists to store categorized fruits
good_fruits = []
bad_fruits = []

small_bananas = []
medium_bananas = []
large_bananas = []

small_apples = []
medium_apples = []
large_apples = []

small_oranges = []
large_oranges = []

for box, label, confidence in zip(boxes, labels, confidences):
    if confidence < confidence_threshold:
        continue  # Skip detections below the confidence threshold

    xmin, ymin, xmax, ymax = map(int, box[:4])
    label_name = classes[label]

    # Calculate the width and height in inches
    width_pixels = xmax - xmin
    height_pixels = ymax - ymin

    width_inches = width_pixels / (new_width / width)
    height_inches = height_pixels / (new_height / height)

    # Check if the fruit is good or bad based on the label
    is_good_fruit = "good" in label_name

    # Categorize bananas
    if "banana" in label_name:
        if banana_small_range[0] <= width_inches <= banana_small_range[1]:
            small_bananas.append((label_name, confidence))
        elif banana_medium_range[0] <= width_inches <= banana_medium_range[1]:
            medium_bananas.append((label_name, confidence))
        elif banana_large_range[0] <= width_inches <= banana_large_range[1]:
            large_bananas.append((label_name, confidence))

    # Categorize apples
    elif "apple" in label_name:
        diameter = max(width_inches, height_inches)
        if apple_small_diameter_range[0] <= diameter <= apple_small_diameter_range[1]:
            small_apples.append((label_name, confidence))
        elif apple_medium_diameter_range[0] <= diameter <= apple_medium_diameter_range[1]:
            medium_apples.append((label_name, confidence))
        elif apple_large_diameter_range[0] <= diameter <= apple_large_diameter_range[1]:
            large_apples.append((label_name, confidence))

    # Categorize oranges
    elif "orange" in label_name:
        if orange_small_range[0] <= width_inches <= orange_small_range[1]:
            small_oranges.append((label_name, confidence))
        elif orange_large_range[0] <= width_inches <= orange_large_range[1]:
            large_oranges.append((label_name, confidence))

    # Categorize fruits as good or bad
    if is_good_fruit:
        good_fruits.append((label_name, confidence))
    else:
        bad_fruits.append((label_name, confidence))

# Display the frame with bounding boxes and labels
for box, label, confidence in zip(boxes, labels, confidences):
    if confidence < confidence_threshold:
        continue  # Skip drawing low-confidence detections

    xmin, ymin, xmax, ymax = map(int, box[:4])
    label_name = classes[label]
    color = (0, 255, 0) if "good" in label_name else (0, 0, 255)  # Green for good, red for bad

    # Draw bounding box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

    # Display class label and confidence score
    text = f"{label_name} ({confidence:.2f})"
    cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the categorized fruits
cv2.putText(frame,
            f"Good Fruits: {', '.join([f'{label} ({confidence:.2f})' for label, confidence in good_fruits])}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(frame, f"Bad Fruits: {', '.join([f'{label} ({confidence:.2f})' for label, confidence in bad_fruits])}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(frame,
            f"Small Bananas: {', '.join([f'{label} ({confidence:.2f})' for label, confidence in small_bananas])}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(frame,
            f"Medium Bananas: {', '.join([f'{label} ({confidence:.2f})' for label, confidence in medium_bananas])}",
            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(frame,
            f"Large Bananas: {', '.join([f'{label} ({confidence:.2f})' for label, confidence in large_bananas])}",
            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(frame,
            f"Small Apples: {', '.join([f'{label} ({confidence:.2f})' for label, confidence in small_apples])}",
            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(frame,
            f"Medium Apples: {', '.join([f'{label} ({confidence:.2f})' for label, confidence in medium_apples])}",
            (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(frame,
            f"Large Apples: {', '.join([f'{label} ({confidence:.2f})' for label, confidence in large_apples])}",
            (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(frame,
            f"Small Oranges: {', '.join([f'{label} ({confidence:.2f})' for label, confidence in small_oranges])}",
            (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(frame,
            f"Large Oranges: {', '.join([f'{label} ({confidence:.2f})' for label, confidence in large_oranges])}",
            (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the frame
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imshow('Fruit Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
