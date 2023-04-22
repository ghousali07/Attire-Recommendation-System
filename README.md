# Attire-Recommendation-System

Artificial Intelligence (AI) has grown significantly, including creating customized shopping experiences, providing personalized ads, and identifying objects and colours from images. Fashion has become an essential aspect of contemporary culture, as people often express themselves and stand out from the crowd through their style choices. This project aims to develop an Attire Recommendation System that utilizes AI to classify clothing items and suggests suitable outfits for specific occasions. The proposed system can process images and live video footage of the user's clothes, analyze the type and colour of each item, and generate outfit recommendations based on the user's existing wardrobe. Users have access to a closet where they can store images of their clothes or use live video recordings. The system uses machine and deep learning techniques to classify clothes and identify colours from images/ videos. An algorithm is proposed for suggesting matching outfits to the user.

## Introduction:

In recent years, the fashion industry has experienced a significant transformation due to technological advancements, particularly in artificial intelligence. The rise of e-commerce platforms and social media has created an enormous demand for personalized recommendations and a seamless shopping experience. This has led to the development of fashion recommendation systems that leverage machine learning and deep learning techniques to analyze users' preferences and provide tailored recommendations. This report will explore the various types of fashion recommendation systems, their working principles, and their impact on the fashion industry. We will also analyze the challenges of developing and implementing these systems and discuss future trends.
Currently, no existing system can suggest clothing options based on the occasion or appropriate colour combinations. For people with a limited fashion sense, selecting an outfit that creates a positive impression can take time and effort. The proposed Attire Recommendation System aims to simplify this process by allowing users to store images of their clothing in a digital wardrobe or record live video footage and receive recommendations for outfits suitable for specific occasions. The primary objective of this system is to suggest the most suitable clothing based on a user's existing wardrobe, eliminating the need to make difficult decisions about what to wear. The system must be accessible, user-friendly, and
capable of handling different types of clothing and colour classification. The wardrobe feature is a crucial aspect of this system, enabling users to manage their uploaded images, and the recommendation algorithm can use this feature to generate suggestions.

## Methodolgy:

We have implemented an attire recommendation system that utilizes ResNet50 for feature extraction and KNN for similarity matching. The ResNet50 model is trained using transfer learning and the extracted features are normalized and stored in pickle files. The KNN algorithm is used to find the closest feature vectors, and the system is deployed using Streamlit, a Python framework that enables the development of interactive web applications. The system provides personalized recommendations based on user preferences, body type, and fashion trends. The report provides detailed information on the model architecture, data preprocessing, and deployment process, making it a valuable resource for researchers and developers interested in building similar recommendation systems.

## Experimentation:

The concept of Transfer learning is used to overcome the issues of the small size Fashion dataset. Therefore we pre-train the classification models on the DeepFashion dataset that consists of 44,441 garment images. The networks are trained and validated on the dataset taken. The training results show a great accuracy of the model with low error, loss and good f-score.

## Libraries Used:
- [OpenCV]() - Open Source Computer Vision and Machine Learning software library

- [Tensorflow]() - TensorFlow is an end-to-end open source platform for machine learning.

- [Tqdm]() - tqdm is a Python library that allows you to output a smart progress bar by wrapping around any iterable.

- [streamlit]() - Streamlit is an open-source app framework for Machine Learning and Data Science teams. Create beautiful data apps in hours, not weeks.

- [pandas]() - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.

- [Pillow]() - PIL is the Python Imaging Library by Fredrik Lundh and Contributors.

- [scikit-learn]() - Scikit-learn is a free software machine learning library for the Python programming language.

- [opencv-python]() - OpenCV is a huge open-source library for computer vision, machine learning, and image processing.

## Conclusion:
Recently, there has been an increasing interest in developing personalized recommendation systems using machine learning and computer vision. Our Attire Recommendation System employed a combination of ResNet50 and KNN algorithms and a large image dataset to generate customized user recommendations. For ease of deployment, we selected Streamlit as our user interface platform. It offered a user-friendly and interactive interface for users to input their preferences and receive real-time recommendations.
Our project has effectively accomplished its objective of delivering a user-friendly and customized solution for attire recommendations. Through the utilization of machine learning algorithms and a large image dataset, we were able to create precise and pertinent recommendations for users. Moreover, the deployment of our project was made efficient by utilizing the Streamlit platform, which enabled users to effortlessly upload images or record live videos and receive personalized recommendations promptly.
The attire recommendation system benefits both customers and sellers in several ways. Firstly, the system provides personalized recommendations for customers based on
their preferences. This can help customers make smarter shopping decisions, discover new clothing items that complement their style, and increase the use of their wardrobe. Secondly, for sellers, the system can assist in selling more products, positively impacting their business. The system can also help users with no fashion sense by recommending the best outfit combinations based on their existing wardrobe. Overall, the attire recommendation system can improve the shopping experience for customers, making it more personalized and enjoyable while also benefiting sellers.
