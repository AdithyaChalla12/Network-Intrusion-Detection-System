# A Hybrid Intrusion Detection System Based on Feature Selection and Weighted Stacking Classifier

## Overview
This project presents a **hybrid Intrusion Detection System (IDS)** designed to improve the accuracy and reliability of detecting cyber-attacks in modern network environments.  
The system combines **feature selection**, **ensemble learning**, and **deep learning models** to address challenges such as high-dimensional data, class imbalance, and false positives.

The proposed approach leverages **Correlation-Based Feature Selection (CFS)** with **Differential Evolution (DE)** and a **Weighted Stacking ensemble classifier** to achieve superior detection performance.

---

## Problem Statement
Traditional intrusion detection systems struggle with:
- High-dimensional network traffic features
- Poor generalization to unseen attacks
- Sensitivity to base classifier selection
- High false positive and false negative rates

This project aims to design an IDS that:
- Improves detection accuracy for both known and unknown attacks
- Reduces false alarms
- Operates effectively on real-world network traffic data

---

## Proposed Solution
The system introduces a **hybrid detection pipeline** consisting of:

1. **Feature Selection**
   - Uses **CFS-DE (Correlation-Based Feature Selection with Differential Evolution)**
   - Reduces feature dimensionality while retaining relevant information

2. **Machine Learning Models**
   - Decision Tree
   - Random Forest
   - Support Vector Classifier (SVC)
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - MLP Classifier

3. **Ensemble Learning**
   - Implements **Weighted Stacking**, where classifiers are weighted based on validation performance
   - Improves robustness over traditional stacking and voting methods

4. **Deep Learning Models**
   - Feed Forward Neural Network (FNN)
   - Recurrent Neural Network (RNN)
   - Captures complex and sequential patterns in network traffic

---

## Dataset
- **CSE-CIC-IDS2018**
- Contains real-world network traffic with multiple attack categories:
  - Denial of Service (DoS)
  - Probe / Surveillance
  - Remote to Local (R2L)
  - User to Root (U2R)
  - Normal traffic

---

## System Architecture
- Data collection and preprocessing
- Feature selection using CFS-DE
- Model training with ML and DL classifiers
- Weighted stacking ensemble
- Evaluation and visualization

---

## Evaluation Metrics
The system is evaluated using standard IDS metrics:
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

### Results
The proposed model achieves:
- **Accuracy:** 99.87%
- **Precision:** 99.88%
- **Recall:** 99.87%
- **F1-score:** 99.88%

These results demonstrate significant improvement over standalone classifiers.

---

## Technologies Used
- **Programming Language:** Python 3
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Deep Learning:** TensorFlow / Keras
- **Visualization:** Matplotlib, Power BI
- **IDE:** Spyder
- **Web Framework (UI):** Flask

---

## Key Features
- Hybrid IDS combining ML and DL models
- Feature dimensionality reduction for faster and accurate classification
- Robust ensemble learning using weighted stacking
- Scalable and adaptable to evolving attack patterns
- Supports real-time intrusion detection scenarios

---

## Future Enhancements
- Integration with real-time network monitoring systems
- Explainable AI (XAI) for improved model interpretability
- Deployment as a cloud-based IDS service
- Support for streaming network traffic data
---

## License
This project is intended for **academic and research purposes**.

