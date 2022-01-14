import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect_faces(our_img):
    new_img = np.array(our_img.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #detect the face
    faces = face_cascade.detectMultiScale(grey,1.1,4)
    #draw a rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img, faces


def detect_eyes(our_img):
    new_img = np.array(our_img.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #detect the eyes
    eyes = eyes_cascade.detectMultiScale(grey,1.2,6)
    #draw a rectangle
    for (x,y,w,h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img


def detect_smiles(our_img):
    new_img = np.array(our_img.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #detect the smile
    smile = smile_cascade.detectMultiScale(grey,1.1,190)
    #draw a rectangle
    for (x,y,w,h) in smile:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

def main():
    """"Face Detection App"""

    st.title("Face Detection App")
    st.text("Built with Streamlit and OpenCV")

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    #effects
    if choice == 'Detection':
        st.subheader("Face Detection")

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original image")
            st.image(our_image)

        enhance_type = st.sidebar.radio("Engance type", ["Original", "Greyscale", "Contrast", "Brightness", "Blurring"])
        if enhance_type == 'Greyscale':
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img,1)
            grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #st.write(new_img)
            st.text("Processed image")
            st.image(grey)
        if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider("Contrast",0.5,3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)
        if enhance_type == 'Brightness':
            c_rate = st.sidebar.slider("Brightness",0.5,3.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)
        if enhance_type == 'Blurring':
            new_img = np.array(our_image.convert('RGB'))
            blur_rate = st.sidebar.slider("Blur",0.5,3.5)
            img = cv2.cvtColor(new_img,1)
            blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
            st.image(blur_img)             

        #detection
        task = ["Faces", "Smiles", "Eyes"]
        feature_choice = st.sidebar.selectbox("Find features",task)
        if st.button("Process"):

            if feature_choice == 'Faces':
                result_img, result_faces = detect_faces(our_image)
                st.image(result_img)

                st.success("Found {} face(s)".format(len(result_faces)))

            elif feature_choice == 'Smiles':
                result_img = detect_smiles(our_image)
                st.image(result_img)

            elif feature_choice == 'Eyes':
                result_img = detect_eyes(our_image)
                st.image(result_img)

    elif choice == 'About':
        st.subheader("About")

if __name__ == '__main__':
    main()