import cv2
import os

# Directory to save captured images
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize the video stream
vs = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = vs.read()
    
    # Display the captured frame
    cv2.imshow("Webcam", frame)

    # Save the frame if 's' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Generate filename
        filename = os.path.join(output_dir, f"captured_image_{len(os.listdir(output_dir)) + 1}.jpg")
        # Save the frame
        cv2.imwrite(filename, frame)
        print(f"Saved image: {filename}")
    # Break the loop if 'q' is pressed
    elif key == ord('q'):
        break

# Release the video stream and close all windows
vs.release()
cv2.destroyAllWindows()
