import cv2
import dlib
import pyautogui

# Load the pre-trained facial landmark model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\AutoScrolling\shape_predictor_68_face_landmarks.dat")

# Start the webcam
cap = cv2.VideoCapture(0)

# Define the threshold line (as a fraction of the frame height)
SCROLL_THRESHOLD = 0.5  # 70% of the frame height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get the coordinates of the eyes
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        
        # Calculate the average vertical position of the eyes
        eye_y = (left_eye[1] + right_eye[1]) / 2

        # Determine the threshold in pixels
        threshold_y = SCROLL_THRESHOLD * frame.shape[0]

        # Check if eyes are below the threshold
        if eye_y > threshold_y:
            pyautogui.scroll(-10)  # Scroll down a little
            
        # Draw threshold line
        cv2.line(frame, (0, int(threshold_y)), (frame.shape[1], int(threshold_y)), (0, 255, 0), 2)
        
        # Highlight eyes for debugging
        cv2.circle(frame, left_eye, 5, (255, 0, 0), -1)
        cv2.circle(frame, right_eye, 5, (255, 0, 0), -1)

    # Display the frame
    cv2.imshow("Eye Tracker", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
