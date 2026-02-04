# 📊 Review Ratings Classifier

## 📝 Project Overview

The **Review Ratings Classifier** is a production-ready Natural Language Processing (NLP) pipeline designed to automate the sentiment analysis of customer reviews. By leveraging supervised machine learning algorithms, the system classifies textual feedback into distinct rating categories, enabling businesses to quantify customer satisfaction and identify areas for improvement at scale.

---

## 🚩 Problem Statement

In the modern e-commerce and service landscape, the volume of user-generated content is overwhelming. Manually reading thousands of reviews to gauge sentiment is inefficient and prone to human bias. This project addresses the need for an automated, scalable, and objective system to categorize reviews based on the underlying sentiment expressed in the text.

---

## 💡 Solution Approach

The project implements a comprehensive NLP pipeline:

1. **Text Preprocessing**: Cleaning noise from raw text (HTML tags, punctuation, stopwords).
2. **Feature Engineering**: Utilizing **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform text into high-dimensional numerical vectors.
3. **Model Benchmarking**: Implementation and comparison of **Multinomial Naive Bayes** and **Support Vector Machines (SVM)** to determine the most effective classifier for the specific dataset.

---

## ✨ Key Features

* **Production-Grade Pipelines**: Clean separation between data preprocessing and model training.
* **Multi-Model Support**: Includes both Naive Bayes (for speed/baseline) and SVM (for high-dimensional accuracy).
* **Text Normalization**: Robust cleaning scripts to handle messy, real-world review data.
* **Model Persistence**: Serialized models (`.pkl`) ready for immediate deployment in API environments.

---

## 🛠 Tech Stack

* **Language**: Python 3.x
* **Libraries**:
* `Scikit-learn`: Model training and feature extraction.
* `Pandas`: Data manipulation and analysis.
* `NumPy`: Numerical operations.
* `NLTK`: Natural language preprocessing tasks.
* `Pickle`: Model serialization.



---

## 🧠 Model Architecture & Algorithm

The project compares two primary architectures:

### 1. Multinomial Naive Bayes

A probabilistic classifier based on Bayes' Theorem. It is highly efficient for text classification where feature independence is assumed, making it an excellent baseline for sentiment tasks.

### 2. Support Vector Machine (SVM)

A discriminative classifier that finds the optimal hyperplane to maximize the margin between classes. Given the high dimensionality of TF-IDF vectors, SVM (specifically with a linear kernel) often provides superior accuracy in sentiment analysis.

---

## 📊 Dataset Description

* **Source**: Customer review dataset containing textual feedback and corresponding labels.
* **Size**: Comprehensive collection of reviews stored in `data.csv`.
* **Preprocessing Steps**:
* Tokenization and Case Normalization.
* Removal of English Stopwords.
* Vectorization using TF-IDF to capture term importance relative to the corpus.



---

## 📈 Evaluation Metrics & Results

The models are evaluated using standard classification metrics:

* **Accuracy**: Overall correctness of the model.
* **Precision/Recall**: Performance across specific sentiment classes.
* **F1-Score**: The harmonic mean of precision and recall, ensuring balance in the classification.

---

## ⚙️ Installation Instructions

Ensure you have Python installed. Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/aKmsdfhjb/REVIEW_RATINGS_CLASSIFIER.git
cd REVIEW_RATINGS_CLASSIFIER

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

---

## 🚀 How to Run / Usage

### Training the Models

To retrain the models or explore the data, run the Jupyter Notebooks:

```bash
jupyter notebook "using SVM.ipynb"
# OR
jupyter notebook "NB.ipynb"

```

### Inference

To use the pre-trained models for classification:

```python
import pickle

# Load the saved SVM model
with open('svm_sentiment_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict sentiment for a new review
prediction = model.predict(["The product quality is excellent and I am very happy!"])
print(f"Predicted Rating/Sentiment: {prediction}")

```

---

## 📂 Project Structure

```text
REVIEW_RATINGS_CLASSIFIER/
├── data.csv                        # Raw dataset
├── NB.ipynb                        # Naive Bayes Implementation
├── using SVM.ipynb                 # SVM Implementation
├── sentiment_analysis_model.pkl    # Serialized NB Model
├── svm_sentiment_svm_model.pkl     # Serialized SVM Model
└── README.md                       # Project Documentation

```

---

## ⚠️ Limitations

* **Sarcasm Detection**: Like most TF-IDF based models, it may struggle with highly nuanced or sarcastic text.
* **Contextual Embeddings**: The current version uses frequency-based vectors rather than contextual embeddings (like BERT), which may limit performance on complex linguistic structures.

---

## 🗺 Future Improvements

* [ ] Implement Deep Learning models (LSTM/GRU).
* [ ] Integrate Transformer-based models (BERT/RoBERTa) for better context understanding.
* [ ] Build a Flask or FastAPI wrapper for real-time inference.
* [ ] Add support for multi-language review classification.

---

## 🤝 Contribution Guidelines

Contributions are welcome! If you'd like to improve the model or add features:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🏗 Acknowledgements

* Scikit-learn Documentation for ML implementation best practices.
* Dataset contributors for providing the raw feedback data.
