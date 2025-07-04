import cv2

img = cv2.imread("test_image.png")
if img is None:
    print(" couldn't read the image.")
else:
    print("Image loaded successfully!")


