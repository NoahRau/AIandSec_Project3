# 🛡️ Next-Gen Network Intrusion Detection System (NIDS)

## 📌 Project Overview
This project focuses on building a machine learning-based Network Intrusion Detection System (NIDS) to secure network infrastructures. By analyzing dumped TCP network packets, the system classifies connections as either **Normal** or malicious (categorized into specific attack types).

The solution moves beyond basic classification by implementing **XGBoost** for high-precision detection and an unsupervised **Isolation Forest** to detect potential "Zero-Day" (unknown) attacks.

## 🎯 Objectives
* **Data Analysis:** Process raw TCP dump data containing connection statistics.
* **Multi-Class Classification:** Classify connections into 5 main categories: `Normal`, `DoS` (Denial of Service), `Probe` (Surveillance), `R2L` (Remote to Local), and `U2R` (User to Root).
* **Granular Prediction:** Extend the model to predict the **41 specific attack types** (e.g., `smurf`, `neptune`, `buffer_overflow`).
* **Anomaly Detection:** Implement an unsupervised layer to flag unknown attack signatures.

## 📂 Dataset Structure
The dataset consists of TCP connection records with 41 features (duration, protocol type, service, flag, etc.). The labels are mapped as follows:

| Category | Description | Examples |
| :--- | :--- | :--- |
| **DoS** | Denial of Service | `syn flood`, `neptune`, `smurf` |
| **R2L** | Remote to Local | `guessing password`, `ftp_write` |
| **U2R** | User to Root | `buffer overflow`, `rootkit` |
| **Probe** | Surveillance/Probing | `port scanning`, `nmap` |
| **Normal** | Standard Traffic | Non-malicious activity |

*Mapping logic provided in `attack2category_map.txt`.*

## ⚙️ Methodology

### 1. Data Preprocessing & Hygiene
* **Duplicate Removal:** Identified and removed duplicate rows to prevent data leakage and bias.
* **Feature Engineering:** Applied **One-Hot Encoding** to categorical features (`protocol_type`, `service`, `flag`) to transform them into a machine-readable format without introducing ordinal bias.
* **Stratified Splitting:** Used stratified sampling to maintain the ratio of rare attacks (like U2R) in both training and testing sets.

### 2. Model Selection: XGBoost
I selected **XGBoost (Extreme Gradient Boosting)** over traditional Random Forests for the following reasons:
* **Performance:** Superior accuracy on structured/tabular data.
* **Regularization:** Built-in L1/L2 regularization to combat overfitting.
* **Speed:** Parallel processing capabilities.

### 3. Advanced: Specific Attack Prediction
I trained a secondary XGBoost model to predict the specific attack label (e.g., `neptune` vs `satan`) rather than just the broad category.
* **Finding:** The impact on accuracy was negligible (**-0.0003** difference).
* **Conclusion:** The model captures the underlying signatures of attacks so well that increasing label granularity does not confuse the decision boundaries.

### 4. Bonus: Zero-Day Detection (Unsupervised)
To simulate a real-world scenario where hackers use new tools, I implemented an **Isolation Forest**.
* **Strategy:** Trained *only* on "Normal" traffic to learn the baseline of "good" behavior.
* **Testing:** Exposed the model to the full dataset (Normal + Attacks).
* **Result:** It successfully identified attacks as anomalies with **~90% accuracy** without ever seeing an attack sample during training.

## 📊 Performance & Results

### XGBoost Category Classification
The model achieved near-perfect performance on the test set.

* **Overall Accuracy:** `99.88%`
* **Precision/Recall:** The model scored `1.00` across almost all categories, with a slight dip in detecting `U2R` attacks (Precision: 0.77), which is expected due to the extreme class imbalance (only 16 samples in support).

```text
              precision    recall  f1-score   support

         dos       1.00      1.00      1.00     13778
      normal       1.00      1.00      1.00     20203
       probe       1.00      0.99      1.00      3497
         r2l       1.00      0.97      0.99       298
         u2r       0.77      0.62      0.69        16

    accuracy                           1.00     37792
```
### Key Insights
* **Top Features:** `src_bytes` (data volume) and specific TCP flags were the most critical indicators of malicious activity.
* **Generalization:** While the model scored ~99% on the KDD99 subset, a stress test on the harder **NSL-KDD** dataset yielded **77.77% accuracy**. This highlights the "Reality Gap" in cybersecurity AI—models trained on older data struggle with distribution shifts in newer network traffic.

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/NIDS-Project.git](https://github.com/yourusername/NIDS-Project.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    Open `solution.ipynb` in Jupyter Notebook or VS Code to reproduce the training and visualization steps.

## 💻 Tech Stack
* **Language:** Python 3.12
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost
* **Visualization:** Matplotlib, Seaborn

## 📬 Contact
**[Your Name]**
[Link to your LinkedIn] | [Link to your Portfolio]