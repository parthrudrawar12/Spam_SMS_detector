import streamlit as st
import pickle
import pandas as pd
import nltk
import string
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Fake SMS Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# DOWNLOAD NLTK RESOURCES
# ==========================================

@st.cache_resource
def load_nltk_resources():
    for resource in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    return set(stopwords.words("english")), PorterStemmer()


stop_words, ps = load_nltk_resources()

# ==========================================
# LOAD MODEL AND VECTORIZER
# ==========================================

@st.cache_resource
def load_model():
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    return tfidf, model


try:
    tfidf, model = load_model()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# ==========================================
# TEXT PREPROCESSING
# ==========================================

def transform_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove punctuation and special characters
    tokens = [
        word for word in tokens
        if word.isalnum()
    ]

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        radial-gradient(
            circle at top left,
            #102d68 0%,
            #06152f 45%,
            #020817 100%
        );
    color: white;
}

/* Remove default top spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Main title */
.main-title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    color: #ffffff;
    margin-bottom: 5px;
    animation: fadeIn 1s ease-in-out;
}

.main-title span {
    color: #ffd43b;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #c7d9ff;
    margin-bottom: 30px;
}

/* Glass effect cards */
.glass-card {
    background: rgba(18, 43, 91, 0.72);
    border: 1px solid rgba(93, 151, 255, 0.45);
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);
    margin-bottom: 20px;
}

/* Text area */
.stTextArea textarea {
    background-color: #102956 !important;
    color: white !important;
    border: 1px solid #527ed5 !important;
    border-radius: 14px !important;
    font-size: 17px !important;
}

/* Gradient buttons */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 13px;
    border: none;
    background: linear-gradient(
        90deg,
        #00c6ff,
        #3977ff,
        #bd35ff
    );
    color: white;
    font-size: 19px;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 22px rgba(0, 198, 255, 0.45);
    color: white;
}

/* Result cards */
.result-safe {
    background: linear-gradient(110deg, #b8ffca, #d9ffe2);
    color: #075b29;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 2px solid #20d86a;
    animation: fadeIn 0.7s ease-in-out;
}

.result-spam {
    background: linear-gradient(110deg, #ffb5c0, #ffe0e5);
    color: #a80018;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 2px solid #ff375f;
    animation: fadeIn 0.7s ease-in-out;
}

/* Feature cards */
.feature-card {
    background: rgba(22, 51, 101, 0.65);
    border: 1px solid #244d91;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
    min-height: 140px;
}

.feature-icon {
    font-size: 35px;
}

.feature-title {
    font-weight: bold;
    color: #ffffff;
    font-size: 16px;
}

.feature-text {
    color: #c7d9ff;
    font-size: 13px;
}

/* Animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(12px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Footer */
.footer {
    text-align: center;
    color: #91a9d9;
    font-size: 14px;
    margin-top: 30px;
    border-top: 1px solid #233e70;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="main-title">
    🛡️ Fake <span>SMS</span> Detection
</div>

<div class="subtitle">
    🟢 Check if your message is Spam or Not Spam
    <br>
    Powered by Natural Language Processing and Machine Learning
</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================
# MAIN LAYOUT
# ==========================================

left_col, right_col = st.columns(
    [2.1, 1],
    gap="large"
)

# ==========================================
# LEFT COLUMN
# ==========================================

with left_col:

    st.markdown("""
    <div class="glass-card">
        <h2>📩 Enter Your Message</h2>
        <p style="color:#c7d9ff;">
            Type or paste the SMS you want to check.
        </p>
    </div>
    """, unsafe_allow_html=True)

    input_sms = st.text_area(
        "SMS Message",
        placeholder=(
            "Example: Congratulations! You have won "
            "a free iPhone. Click here to claim your prize!"
        ),
        height=150,
        max_chars=500,
        label_visibility="collapsed"
    )

    st.caption(f"Characters: {len(input_sms)}/500")

    predict_clicked = st.button(
        "🚀  Predict Message",
        use_container_width=True
    )

    # ======================================
    # PREDICTION
    # ======================================

    if predict_clicked:

        if not input_sms.strip():

            st.warning("⚠️ Please enter an SMS message first.")

        else:

            with st.spinner("🤖 Analyzing your message..."):

                try:
                    transformed_sms = transform_text(input_sms)

                    vector_input = tfidf.transform(
                        [transformed_sms]
                    )

                    result = model.predict(vector_input)[0]

                    st.session_state["prediction"] = result
                    st.session_state["original_sms"] = input_sms

                except Exception as e:

                    st.error(
                        f"Prediction failed. Check your model "
                        f"and preprocessing settings. Details: {e}"
                    )

    # ======================================
    # DISPLAY RESULT
    # ======================================

    if "prediction" in st.session_state:

        result = st.session_state["prediction"]

        st.write("")

        # Assumes your model uses 1 = spam and 0 = not spam.
        if result == 1:

            st.markdown("""
            <div class="result-spam">
                <div style="font-size:55px;">🚨</div>
                <h1>Spam Message!</h1>
                <h3>⚠️ This message may be suspicious.</h3>
                <p>Be careful with links and requests for personal details.</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-safe">
                <div style="font-size:55px;">🛡️</div>
                <h1>Not Spam!</h1>
                <h3>✅ This message appears legitimate.</h3>
                <p>Always verify unexpected requests independently.</p>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "This is a machine-learning prediction, "
            "not a guarantee that the message is safe."
        )

        if st.button("🔄 Clear Result"):

            del st.session_state["prediction"]
            st.rerun()

    # ======================================
    # PROCESSING PIPELINE
    # ======================================

    st.write("")

    st.markdown("""
    <div class="glass-card">
        <h3 style="text-align:center;">
            ⚙️ How the System Works
        </h3>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚙️</div>
            <div class="feature-title">1. Preprocessing</div>
            <div class="feature-text">
                Lowercase, tokenize, remove stopwords and stem.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔤</div>
            <div class="feature-title">2. Vectorization</div>
            <div class="feature-text">
                Convert text into TF-IDF numerical features.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">3. Prediction</div>
            <div class="feature-text">
                Analyze features using the trained model.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🛡️</div>
            <div class="feature-title">4. Result</div>
            <div class="feature-text">
                Display Spam or Not Spam.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# RIGHT COLUMN
# ==========================================

with right_col:

    st.markdown("""
    <div class="glass-card">
        <h3>💡 Try Example Messages</h3>
        <p style="color:#c7d9ff;">
            Copy a sample and paste it into the message box.
        </p>
    </div>
    """, unsafe_allow_html=True)

    examples = [
        (
            "🎁 Lottery Prize",
            "Congratulations! You won ₹25,00,000. "
            "Send your bank details to claim your prize."
        ),
        (
            "🏦 Fake Bank Alert",
            "URGENT! Your bank account will be blocked. "
            "Click here to verify your KYC immediately."
        ),
        (
            "👋 Personal Message",
            "Hi, I will reach the college in 15 minutes. "
            "Please wait near the main gate."
        ),
        (
            "📚 Class Reminder",
            "Reminder: Your Python class starts tomorrow "
            "at 10 AM. Please bring your notebook."
        )
    ]

    for title, message in examples:

        with st.container(border=True):

            st.markdown(f"**{title}**")

            st.caption(message)

            if st.button(
                "Use This Message",
                key=title
            ):
                st.session_state["example_sms"] = message
                st.session_state.pop("prediction", None)
                st.rerun()

    # About section

    st.markdown("""
    <div class="glass-card">
        <h3>✨ Features</h3>
        <p>✅ Text preprocessing</p>
        <p>✅ TF-IDF vectorization</p>
        <p>✅ Machine learning prediction</p>
        <p>✅ Interactive Streamlit UI</p>
        <p>✅ Anipythonmated prediction results</p>
        <p>✅ Real-time classification</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>🎯 Stay Aware. Stay Safe!</h3>
        <p style="color:#c7d9ff;">
            Never share OTPs, passwords, or banking details
            with unknown senders.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class=
"footer">
    Built with ❤️ using Python, Streamlit, NLP and Machine Learning
    <br><br>
    Fake SMS Detection | Data Science Portfolio Project
</div>
""", unsafe_allow_html=True)


# copy this and run it
#  python-m streamlit run prediciton.py
