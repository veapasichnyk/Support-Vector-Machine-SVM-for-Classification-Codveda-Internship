# 📌 Churn Prediction with Support Vector Machines (SVM)

A machine learning project to predict **customer churn** using **Support Vector Machines (SVM)** on the **Telecom Churn dataset**.  

---

## 📊 Dataset
- **Rows:** *3,333*  
- **Features:** *20 (numeric + categorical)*  
- **Target:** *Churn (binary: Yes/No)*  
- Includes:  
  - 📞 *Service plan info* (`International plan`, `Voice mail plan`)  
  - 📊 *Usage statistics* (day, evening, night, international calls/minutes/charges)  
  - ☎️ *Customer service calls*  

---

## ⚙️ Features
- 🔄 *Preprocessing*: encoding categorical variables + scaling numeric features  
- ✂️ *Stratified split*: train / validation / test  
- 🤖 *Models*: SVM with **Linear** and **RBF kernels**  
- ⚖️ *Class balancing*: `class_weight="balanced"`  
- 📏 *Metrics*: Accuracy, Precision, Recall, F1, AUC  
- 📉 *Visualization*: ROC Curves, Decision Boundaries, Precision–Recall tradeoff  

---

## 📈 Example Results
- **Accuracy:** ~74%  
- **F1 (churn class):** ~0.46  
- **Recall (churn class):** ~0.76  
- **Test AUC:** ~0.82  
- **Top features (Linear SVM):** `Customer service calls`, `Total day minutes`, `International plan`  

---

## 🛠 Tech Stack
- 🐍 *Python 3.12+*  
- 📚 *scikit-learn, pandas, numpy*  
- 🎨 *matplotlib, seaborn*  

---

## 📜 License
- **MIT License** — free to use and adapt.  
