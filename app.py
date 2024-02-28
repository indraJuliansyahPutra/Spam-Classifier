import streamlit as st
import pickle
import base64

def set_background(image_file):
    with open(image_file, 'rb') as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
    <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            color: white;
            text-align: center;
        }}
        .stMarkdown {{
            color: white;
        }}
        .error-message {{
            background-color: #ff0000;  /* Merah untuk pesan kesalahan */
            color: white;
            padding: 10px;
            border-radius: 5px;
            display: inline-block;
        }}

        .success-message {{
            background-color: #00ff00;  /* Hijau untuk pesan berhasil */
            color: white;
            padding: 10px;
            border-radius: 5px;
            display: inline-block;
        }}
    </style>
"""



    st.markdown(style, unsafe_allow_html=True)

cv = pickle.load(open('dataset/cv_terbaik.pkl', 'rb'))
clf = pickle.load(open('dataset/clf_terbaik.pkl', 'rb'))

def predict_email(input_text):
    input_vector = cv.transform([input_text])
    prediction = clf.predict(input_vector)
    probs = clf.predict_proba(input_vector)

    return prediction[0], probs[0]

set_background('./images/background.jpg')
st.title('Email Spam Classification')
st.write('Enter the email text below to check if it\'s spam or not.')

input_email = st.text_area('Enter Email Text: ', value='', height=200)

if st.button('Predict'):
    result, confidence = predict_email(input_email)

    if result == 1:
        st.markdown('<p class="error-message">This email is spam</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="success-message">This Email is not spam</p>', unsafe_allow_html=True)

    st.write('Confidence Score: ')
    st.write('Spam Score: {:.2%}'.format(confidence[1]))
    st.write('Not Spam Score: {:.2%}'.format(confidence[0]))