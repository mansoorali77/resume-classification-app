import streamlit as st
import pickle

# Load the model and vectorizer
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
vectorizer = model_data["vectorizer"]

st.title("Resume Classification App")

uploaded_file = st.file_uploader("Upload a resume (PDF/DOCX)", type=["pdf", "docx"])

def extract_text_from_file(file):
    import PyPDF2
    import docx
    import tempfile

    text = ""
    file_ext = file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    if file_ext == "pdf":
        try:
            with open(tmp_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    elif file_ext == "docx":
        try:
            doc = docx.Document(tmp_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")

    return text

if uploaded_file:
    raw_text = extract_text_from_file(uploaded_file)

    if raw_text:
        import re
        cleaned = re.sub(r"[^\w\s]", " ", raw_text.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Vectorize and predict
        X = vectorizer.transform([cleaned])
        prediction = model.predict(X)[0]
        st.success(f"Predicted Job Category: **{prediction}**")
