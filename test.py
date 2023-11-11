import firebase_admin
from firebase_admin import credentials, db, storage
import cv2
import time

# Fetch the service account key JSON file contents
cred = credentials.Certificate('firebase_config.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://riri-project-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'riri-project.appspot.com'
})

bucket = storage.bucket()
ref = db.reference('images')

cam = cv2.VideoCapture(2)

cv2.namedWindow("test")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        timestamp = int(time.time())
        img_name = "image_{}.jpg".format(timestamp)
        cv2.imwrite(img_name, frame)

        # Upload image to Firebase Storage
        
        blob = bucket.blob(img_name)
        with open(img_name, "rb") as img_file:
            blob.upload_from_file(img_file)

        # Get the download URL of the uploaded image
        img_url = blob.public_url

        # Send URL to Realtime Database
        ref.child(str(timestamp)).set({
            'timestamp': timestamp,
            'image_url': img_url,
            'img_name':img_name

        })

        print("Image URL {} uploaded to Storage and added to RTDB!".format(img_url))

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
