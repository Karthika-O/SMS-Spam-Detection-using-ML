import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to classify SMS
def classify_sms(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return "Spam" if prediction == 1 else "Not Spam"

# Custom CSS for styling background image on the home page
st.markdown(
    """
    <style>

    .title {
        font-size: 48px;
        color: #ff4b4b;
        font-weight: bold;
        text-align: center;
    }
    .subheader {
        font-size: 24px;
        text-align: center;
        color: #4b86b4;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 14px;
        margin-top: 20px;
    }
    .stTextInput > label {
        font-size: 18px;
    }
    .result {
        font-size: 24px;
        color: white;
        background-color: #4CAF50;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar Menu
st.sidebar.title("SMS SPAM DETECTION")
#st.sidebar.header("Navigate")
menu = st.sidebar.selectbox("", ["Home", "About Project", "About the Model"])

# Main Page Content based on the menu selection
if menu == "Home":
    # Only show background image for Home page
    st.markdown("<div class='main'></div>", unsafe_allow_html=True)

    # Main UI with title and input
    st.markdown("<h1 class='title'>SMS Spam Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Enter your message below to see if it's spam or not</p>", unsafe_allow_html=True)

    # Input section for Home page
    user_input = st.text_area("📨 Type your SMS message here", "")
    # Classify button with an image
    if st.button("🔍 Classify Message"):
        if user_input:
            result = classify_sms(user_input)
            result_color = "#ff4b4b" if result == "Spam" else "#4CAF50"
            # Display the result with a background image
            result_image = "spam.webp" if result == "Spam" else "not-spam.jpg"  # Replace with your own images
            st.image(result_image, use_column_width=True)
            st.markdown(f"<div class='result' style='background-color:{result_color};'>{result}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a message before classifying.")

elif menu == "About Project":
    st.title("About the Project")
    st.write("""
        This SMS Spam Detection app uses machine learning to classify SMS messages as either 'Spam' or 'Not Spam'.
        The project aims to provide users with a quick and easy way to check if a message is likely to be spam.
        The underlying model has been trained on a dataset of labeled SMS messages using advanced machine learning techniques.
    """)

elif menu == "About the Model":
    st.title("About the Model")
    st.write("""
        The model used in this project is a machine learning classifier trained on SMS data. It leverages text features extracted using a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer, 
        which converts the SMS messages into a numerical form that can be used by machine learning algorithms.
        
        The model has been trained using a combination of Naive Bayes and Support Vector Machine (SVM) algorithms, which have shown high accuracy in detecting spam.
        It evaluates messages based on patterns found in historical SMS datasets.
    """)

# Footer with logo on all pages
#footer_logo = Image.open("footer_logo.png")  # Replace with your own logo
#st.markdown("<p class='footer'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
#st.image(footer_logo, width=50)
