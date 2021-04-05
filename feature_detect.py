import numpy as np
import cv2 as cv
import math
import random
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import imutils as im
from imutils import face_utils
import dlib

# Mohammad H. Mansoor

im1 = cv.imread('input_image.jpg')
plt.figure()
img = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
plt.title('Original')
plt.imshow(img,cmap='gray', vmin=0, vmax=255)

# Apply average filter
box = np.array(
    [[0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1]]
)

average = cv.filter2D(img,-1,box)    
plt.figure()
plt.title("Average Filter")
plt.imshow(average,cmap='gray', vmin=0, vmax=255)

sobel_vert = np.array([
         [-1.0, 0.0, 1.0]
        ,[-2.0, 0.0, 2.0]
        ,[-1.0, 0.0, 1.0]
        ])
sobel_horiz = sobel_vert.T

d_horiz = convolve2d(img, sobel_horiz, mode='same', boundary = 'symm', fillvalue=0)
d_vert = convolve2d(img, sobel_vert, mode='same', boundary = 'symm', fillvalue=0)
grad=np.sqrt(np.square(d_horiz) + np.square(d_vert))
grad *= 255.0 / np.max(grad)
plt.figure()
plt.title('Gradient Edge by 2d Conv')
plt.imshow(grad,cmap='gray', vmin=0, vmax=255)

# Find the faces in the image
cascPath = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv.CascadeClassifier(cascPath)

faces = faceCascade.detectMultiScale(
    img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

print("Found {0} face(s)!".format(len(faces)))

# Draw a square around the face
for (x, y, w, h) in faces:
    cv.rectangle(im1, (x, y-40), (x+w, y+h+80), (0, 255, 0), 2)
    cut =im1[y-40:y+h+80,x:x+w,:] 
    cv.imwrite("face_detected.png",cut)
    plt.figure()
    plt.title('face detected.')
    plt.imshow(cv.cvtColor(im1,cv.COLOR_BGR2RGB))

# Display the cut out image
face_cut_out = cv.imread('face_detected.png')
plt.figure()
plt.title('Face cut out')
plt.imshow(cv.cvtColor(face_cut_out,cv.COLOR_BGR2RGB))

detector = dlib.get_frontal_face_detector()

face = detector(face_cut_out, 1)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Links referenced for this code:
# https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
# https://www.geeksforgeeks.org/opencv-facial-landmarks-and-face-detection-using-dlib-and-opencv/
# https://livecodestream.dev/post/detecting-face-features-with-python/
for (i, faces) in enumerate(face):
    shape = predictor(face_cut_out, faces)
    shape = face_utils.shape_to_np(shape)
    (x,y,w,h) = face_utils.rect_to_bb(faces)

    for (x, y) in shape:
        cv.circle(face_cut_out, (x,y),3, (255, 0, 0), -1)

plt.figure()
plt.title('Final image with facial landmarks')
plt.imshow(cv.cvtColor(face_cut_out, cv.COLOR_BGR2RGB))



plt.show()