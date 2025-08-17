# 📧 Spam Email Classification using Machine Learning  

Hi, I’m Ajay 👋  
This repository contains my **Spam Email Classification project**, where I built and compared multiple ML models to detect whether an email is **Spam or Not Spam (Ham)**.  

This project was inspired by the **CampusX Spam Classification project**, but I extended it with my own feature engineering, model comparison, and performance benchmarking.  

---

## 🔍 Project Overview  
Email spam is a huge problem, with billions of unwanted emails sent daily. To tackle this, I:  
- Collected and preprocessed a labeled email dataset.  
- Cleaned the text (removing punctuation, stopwords, special characters).  
- Used **TF-IDF vectorization** to represent text numerically.  
- Trained multiple models including **Naive Bayes, Random Forest, Extra Trees, and XGBoost**.  
- Evaluated them on **precision, recall, F1-score, and accuracy**.  

---

## 🛠️ Tech Stack  
- **Python**  
- **Pandas, NumPy** – data preprocessing  
- **Scikit-learn** – ML models, TF-IDF vectorization, metrics  
- **XGBoost** – advanced boosting model  
- **Matplotlib, Seaborn** – visualization  

---

## 📊 Model Performance  

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Naive Bayes        | 94.10%   | 100%      | 92%    | 95.8%    |
| Random Forest      | 96.8%    | 97%       | 96%    | 96.5%    |
| Extra Trees        | 97.2%    | 97%       | 97%    | 97%      |
| **XGBoost (Final)**| **98.3%**| **98%**   | **98%**| **98%**  |

✨ The **XGBoost model** was chosen as the final model because it gave the best balance of precision and recall, ensuring fewer false positives and negatives.  

---

## 🌟 Key Insights  
- **Naive Bayes** is simple and surprisingly effective for text classification.  
- **Ensemble models (Random Forest, Extra Trees, XGBoost)** consistently outperform simpler models.  
- Precision matters a lot — flagging a real email as spam is more harmful than missing a spam mail.  
- Feature engineering (like removing rare words and normalizing text) significantly improves results.  

---

## 🚀 Future Improvements  
- Build a **Flask/Streamlit app** for real-time email classification.  
- Deploy the model as an **API**.  
- Add support for **multi-language spam detection**.  
- Train on larger, real-world datasets (Enron, SpamAssassin).  

---
## 📬 Connect with Me

LinkedIn: linkedin.com/in/budida-ajay

GitHub: github.com/ajay9704


----
## 📌 How to Run  
```bash
# Clone the repository
git clone https://github.com/ajay9704/spam-classification.git

# Navigate into the project
cd spam-classification

# Install dependencies
pip install -r requirements.txt




# Run the model training
python main.py
