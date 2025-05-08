# 🚀 Logistic Regression from Scratch in Python

This project demonstrates how to build a logistic regression classifier **from scratch** using only NumPy—no machine learning libraries like scikit-learn were used for the algorithm itself. The goal is to understand and implement the underlying math, including the **sigmoid function**, **gradient descent**, and **cross-entropy (log) loss**.

---

## 📚 What This Project Covers

- Logistic Regression fundamentals
- Implementation of the sigmoid activation function
- Loss function: Cross Entropy (Log Loss)
- Manual gradient descent optimization
- Accuracy evaluation on a real-world dataset (PIMA Indian Diabetes dataset)

---

## 🧠 Math Behind It

The logistic regression algorithm is built upon:

- **Sigmoid function**:  
  \[
  \sigma(x) = \frac{1}{1 + e^{-x}}
  \]

- **Cross-Entropy Loss**:  
  \[
  L = -\frac{1}{N} \sum \left[y \log(p) + (1 - y) \log(1 - p)\right]
  \]

- **Gradient Descent** for updating weights and bias.

---

## 🧪 Dataset Used

- [PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 🛠️ Libraries Used

Only essential libraries were used:

```bash
numpy
pandas
scikit-learn (for data preprocessing and train-test split)
