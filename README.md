# 🏥 Adverse Drug Reaction (ADR) Prediction System

An AI-powered system that predicts potential adverse drug reactions (side effects) based on a selected **drug** and **clinical indication** using **Graph Neural Networks (RGCN)** and Machine Learning techniques.

---

## 🚀 Project Overview

Adverse Drug Reactions (ADRs) can be harmful and sometimes life-threatening. This project aims to:

- Predict possible side effects of drugs  
- Provide **probability** and **confidence scores**  
- Assist in **drug safety analysis**  

The system was developed using two approaches:

- 📊 **Stacking Machine Learning Model** (XGBoost, LightGBM, CatBoost + Meta Model) — used for experimentation and performance comparison  
- 🌐 **Graph Neural Network (RGCN)** — used as the **final deployed model** for prediction  

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
- **Machine Learning Models (Experimental)**:
  - XGBoost
  - LightGBM
  - CatBoost
  - Random Forest (Meta Model)
- **Final Model (Deployed)**:
  - RGCN (Relational Graph Convolutional Network - PyTorch)
- **Libraries**:
  - Pandas, NumPy
  - Scikit-learn
  - PyTorch
  - Plotly

---

## 📂 Dataset

The project uses multiple datasets:

- `drug_names.tsv` → Drug IDs and names  
- `meddra_all_indications.tsv` → Drug–Indication relationships  
- `meddra_all_se.tsv` → Drug–Side Effect relationships  
- `drugsComTrain/Test.csv` → Real-world drug-related data  

These datasets are merged to create:
Drug → Indication → Side Effect



This unified dataset is used for both machine learning and graph-based modeling.

---

## ⚙️ How It Works

### 🔹 Step 1: User Input
- Select Drug  
- Select Clinical Indication  

---

### 🔹 Step 2: Data Processing
- Load dataset  
- Filter relevant side effects  
- Clean and normalize text  
- Build relationships between drug, indication, and side effects  

---

### 🔹 Step 3: Model Development

#### 🟢 Stacking Model (Experimental Phase)
- Base models:
  - XGBoost
  - LightGBM
  - CatBoost  
- Meta model (Random Forest) combines predictions  
- Used for:
  - Performance comparison  
  - Evaluation of feature-based learning  

---

#### 🔵 RGCN Model (Final Deployed Model)

- Data is converted into a **graph structure**
- Nodes:
  - Drug
  - Indication
  - Side Effect  
- Edges:
  - Drug → Indication  
  - Drug → Side Effect  

---

### 🔹 Step 4: Prediction (RGCN)

1. Convert selected inputs into node IDs  
2. Generate **node embeddings** using RGCN layers  
3. Combine embeddings:
   - Drug embedding  
   - Indication embedding  
   - Side effect embedding  
4. Pass through neural network scorer  
5. Apply **sigmoid function** to get probability  

---

### 🔹 Step 5: Output

For each side effect:

- **Probability** → likelihood of occurrence  
- **Confidence** → reliability of prediction  

---

## 📊 Evaluation Metrics

The models were evaluated using:

- **Precision**
- **Recall**
- **F1 Score**

### 🔹 Stacking Model Performance (Experimental Benchmark)

- Precision ≈ 0.91  
- Recall ≈ 0.82  
- F1 Score ≈ 0.86  

---

### 🔹 RGCN Model (Final Model)

- Evaluated using the same metrics  
- Focused on capturing **graph-based relationships**  
- Produces strong probability scores based on connectivity patterns  

---

## 📈 Visualizations

- Bar Chart → Top side effects  
- Scatter Plot → Probability vs Confidence  
- Donut Chart → Average Confidence  

These visualizations help users interpret predictions effectively.

---

## ⚠️ Limitations

- Trained on subset (~500K rows)  
- No patient-specific factors (age, dosage, medical history)  
- Negative samples generated via sampling  
- RGCN may produce **high probability values** due to dense graph connections  
- Confidence in RGCN is rule-based, not learned  

---

## 🔮 Future Work

- Train on full dataset with higher computational power  
- Add drug–drug interaction modeling  
- Improve probability calibration  
- Integrate real-world clinical data  
- Use advanced NLP models (BERT) for text features  
- Improve confidence estimation using uncertainty modeling  

---

## 💡 Key Learnings

- Graph-based models effectively capture relationships between entities  
- Ensemble models provide stable and balanced predictions  
- Different modeling approaches give different perspectives on the same data  
- Combining approaches improves overall understanding of the problem  

---

## 🏁 Conclusion

This project demonstrates how AI and graph-based learning can be used to predict adverse drug reactions by capturing relationships between drugs, indications, and side effects.

The RGCN model serves as the final deployed model, while stacking models provide a strong comparative baseline. The system provides interpretable insights through an interactive dashboard and can be extended for real-world healthcare applications.

---

## 📸 Demo

👉 Streamlit App:  
https://adverse-drug-reactionpredictionsystem-zcapb6bza78xtaqriz96nv.streamlit.app/

---

## 👤 Author

**Srinadh Doppalapudi**  
MS in Data Analytics  

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
