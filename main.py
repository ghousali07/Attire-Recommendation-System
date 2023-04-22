import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2
from streamlit_option_menu import option_menu
from pathlib import Path
import streamlit_authenticator as stauth


st.set_page_config(page_title="Recommender System", page_icon=":bar_chart:", layout="wide")

# --- USER AUTHENTICATION ---
names = ["Yash Patel", "Tarun Mukesh", "M Ghous", "M Faraaz", " Naveen", "K Bhatt"]
usernames = ["yash", "tarun", "ghous", "faraaz", "naveen", "bhatt"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "recommender", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")
    
if authentication_status:

# -- Sidebar --
    authenticator.logout("Logout", "main")
    selected2 = option_menu(None, ["Home", "About Us", "Contact"], 
        icons=['house', 'file-person', "person-lines-fill"], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    selected2

    # -- Main Page --

    inf =  False

    st.title('Attire Recommender System')

    feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
    filenames = pickle.load(open('filenames.pkl','rb'))

    model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
    model.trainable = False

    model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    def save_uploaded_file(uploaded_file):
        try:
            with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
                f.write(uploaded_file.getbuffer())
            return 1
        except:
            return 0

    def feature_extraction(img_path,model):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        return normalized_result

    def recommend(features,feature_list):
        neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)

        distances, indices = neighbors.kneighbors([features])

        return indices

    def show_image(image):
            # feature extract
            features = feature_extraction(image, model)
            indices = recommend(features,feature_list)
            # show
            col1,col2,col3,col4,col5 = st.columns(5)

            with col1:
                st.image(filenames[indices[0][0]])
                st.button("Try Now", key="1")
            with col2:
                st.image(filenames[indices[0][1]])
                st.button("Try Now", key="2")
            with col3:
                st.image(filenames[indices[0][2]])
                st.button("Try Now", key="3")
            with col4:
                st.image(filenames[indices[0][3]])
                st.button("Try Now", key="4")
            with col5:
                st.image(filenames[indices[0][4]])
                st.button("Try Now", key="5")
            return  True
        
    cap = cv2.VideoCapture(0)
        
    st.subheader('Capture or Upload your Product')
    capture = st.checkbox('Capture image')

    if capture:
        capst = st.button('Start Video')
        st.markdown('---')
        stframe = st.empty()
        while capst:
            ret, frame = cap.read()
            if frame is not None:   
                frame2 = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow('frame',frame2)
                stframe.image(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.imwrite('capture.jpg',frame)
                    cap.release()
                    cv2.destroyAllWindows()
                    print('Execution completed')
                    inf = True
                    if  show_image('capture.jpg'):
                        st.success('Recommendation completed')
                    break
            else:
                st.error('No camera found')
                break
                
    else:
        uploaded_file = st.file_uploader("Choose an image")
        if uploaded_file is not None:
            if save_uploaded_file(uploaded_file):
                # display the file
                display_image = Image.open(uploaded_file)
                size = (400,400)
                resize_im = display_image.resize(size)
                st.image(resize_im)
                features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
                indices = recommend(features,feature_list)
                st.subheader("Recommended Products")
                col1,col2,col3,col4,col5 = st.columns(5)
                with col1:
                    st.image(filenames[indices[0][0]])
                    st.button("Try Now", key="6")
                with col2:
                    st.image(filenames[indices[0][1]])
                    st.button("Try Now", key="7")
                with col3:
                    st.image(filenames[indices[0][2]])
                    st.button("Try Now", key="8")
                with col4:
                    st.image(filenames[indices[0][3]])
                    st.button("Try Now", key="9")
                with col5:
                    st.image(filenames[indices[0][4]])
                    st.button("Try Now", key="10")
            else:
                st.header("Some error occured in file upload")
