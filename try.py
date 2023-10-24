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

# Define the confidence threshold for classification
confidence_threshold = 0.8  # Minimum confidence score to consider a detection

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

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

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

    for i in range(len(boxes)):
        box = boxes[i]
        label = classes[labels[i]]
        confidence = confidences[i]

        # Filter detections based on confidence threshold
        if confidence > confidence_threshold:
            x1, y1, x2, y2 = box[:4].astype(int)

            # Calculate the size of the fruit in inches
            width_pixels = x2 - x1
            width_inches = (width_pixels / new_width) * 9  # Assuming a max width of 9 inches

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
            cv2.putText(frame, f'{label} ({width_inches:.2f} inches, {size_category})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Fruit Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
