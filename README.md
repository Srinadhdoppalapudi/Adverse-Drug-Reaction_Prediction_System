# Adverse-Drug-Reaction_Prediction_System

# 🏥 Adverse Drug Reaction (ADR) Prediction System

An AI-powered system that predicts potential adverse drug reactions (side effects) based on a selected **drug** and **clinical indication** using **Machine Learning** and **Graph Neural Networks (RGCN)**.

---

## 🚀 Project Overview

Adverse Drug Reactions (ADRs) can be harmful and sometimes life-threatening. This project aims to:

- Predict possible side effects of drugs  
- Provide **probability** and **confidence scores**  
- Assist in **drug safety analysis**  

The system uses both:
- 📊 **Stacking Machine Learning Model** (XGBoost, LightGBM, CatBoost + Meta Model)  
- 🌐 **Graph Neural Network (RGCN)** for relationship-based learning  

---

## 🧠 Key Features

- 🔍 Select **Drug & Indication**
- ⚠️ Predict **Side Effects**
- 📈 View **Probability & Confidence**
- 📊 Interactive Visualizations
- 📋 Detailed Prediction Table
- 📥 Export results as CSV

---

## 🖥️ Tech Stack

- **Frontend/UI**: Streamlit  
- **Backend**: Python  
- **ML Models**:
  - XGBoost
  - LightGBM
  - CatBoost
  - Random Forest (Meta Model)
- **Deep Learning**:
  - RGCN (PyTorch)
- **Libraries**:
  - Pandas, NumPy
  - Scikit-learn
  - Plotly

---

## 📂 Dataset

The project uses multiple datasets:

- `drug_names.tsv` → Drug IDs and names  
- `meddra_all_indications.tsv` → Drug–Indication relationships  
- `meddra_all_se.tsv` → Drug–Side Effect relationships  
- `drugsComTrain/Test.csv` → Real-world drug data  

These datasets are merged to create:
Drug → Indication → Side Effect




---

## ⚙️ How It Works

### 🔹 Step 1: User Input
- Select Drug  
- Select Clinical Indication  

---

### 🔹 Step 2: Data Processing
- Load dataset  
- Filter relevant side effects  
- Generate features (TF-IDF, encoding, engineered features)

---

### 🔹 Step 3: Prediction

#### 🟢 Stacking Model
- Base models:
  - XGBoost
  - LightGBM
  - CatBoost
- Meta model combines predictions

#### 🔵 RGCN Model
- Graph-based learning
- Nodes:
  - Drug
  - Indication
  - Side Effect
- Learns relationships using embeddings

---

### 🔹 Step 4: Output

For each side effect:
- **Probability** → likelihood of occurrence  
- **Confidence** → reliability of prediction  

---

## 📊 Evaluation Metrics

The models are evaluated using:

- **Precision**
- **Recall**
- **F1 Score**

👉 Meta Model Performance:
- Precision ≈ 0.91  
- Recall ≈ 0.82  
- F1 Score ≈ 0.86  

---

## 📈 Visualizations

- Bar Chart → Top side effects  
- Scatter Plot → Probability vs Confidence  
- Donut Chart → Average Confidence  

---

## ⚠️ Limitations

- Trained on subset (~500K rows)  
- No patient-specific factors (age, dosage, history)  
- Negative samples generated via sampling  
- RGCN may produce high probabilities due to dense graph connections  

---

## 🔮 Future Work

- Train on full dataset  
- Add drug–drug interaction modeling  
- Improve probability calibration  
- Integrate real-world clinical data  
- Use advanced NLP models (BERT)

---

## 💡 Key Learnings

- Graph models capture relationships effectively  
- Ensemble models provide stable predictions  
- Combining approaches improves system robustness  

---

## 🏁 Conclusion

This project demonstrates how AI and machine learning can support **drug safety analysis** by predicting adverse reactions and providing interpretable insights through an interactive dashboard.

---

## 📸 Demo

👉 Streamlit App: https://adverse-drug-reactionpredictionsystem-zcapb6bza78xtaqriz96nv.streamlit.app/

