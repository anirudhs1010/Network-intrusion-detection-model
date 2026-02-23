# Network Intrusion Detection Model

This repository contains a machine learning pipeline designed to classify network connections as either normal traffic or anomalous intrusions. The system is built to execute entirely within a browser-based Jupyter sandbox.

## Dataset Overview

### Background
The dataset to be audited was provided which consists of a wide variety of intrusions simulated in a military network environment. It created an environment to acquire raw TCP/IP dump data for a network by simulating a typical US Air Force LAN. The LAN was focused like a real environment and blasted with multiple attacks. A connection is a sequence of TCP packets starting and ending at some time duration between which data flows to and from a source IP address to a target IP address under some well-defined protocol. Also, each connection is labelled as either normal or as an attack with exactly one specific attack type. Each connection record consists of about 100 bytes.

For each TCP/IP connection, 41 quantitative and qualitative features are obtained from normal and attack data (3 qualitative and 38 quantitative features). The class variable has two categories:
* Normal
* Anomalous

### Data Ingestion
To accommodate browser-based execution without manual file uploads, the data is fetched directly from a public GitHub mirror during runtime:
* `Train_data.csv`: Used for training and validation.
* `Test_data.csv`: Used for generating final unlabelled predictions.

---

## Instructions

### Environment Setup
This pipeline is optimized for browser-based Python environments like JupyterLite or Try Jupyter (Pyodide). Standard local Python execution requires modifying the data ingestion step to use standard local file paths or standard `requests` instead of `pyodide.http`.

1. Open a Pyodide-supported Jupyter environment.
2. Ensure the following packages are available in your environment: `pandas`, `scikit-learn`, `matplotlib`. 
3. Create a new notebook and paste the provided unified pipeline script into a single cell.

### Execution
Run the cell. The script handles the entire workflow autonomously:
1. Fetches the Kaggle dataset directly via HTTPS using `pyodide.http.open_url`.
2. Preprocesses and scales the data.
3. Trains the Random Forest model.
4. Outputs the classification report.
5. Generates matplotlib visualizations for class distribution, protocol usage, and feature importance.

---

## Architecture Analysis

### Preprocessing
The preprocessing pipeline implements strict feature alignment. Categorical variables are transformed via one-hot encoding, and `align` is used to ensure the test set matrix perfectly matches the training set matrix dimensions, mitigating errors from missing categorical levels in the test split. Continuous variables, which vary wildly in magnitude (e.g., byte counts vs. error rates), are normalized using `StandardScaler` to ensure the algorithm evaluates feature variance uniformly.

### Model Selection
A `RandomForestClassifier` (100 estimators) is utilized. For tabular datasets with a mixture of categorical and continuous features, tree-based ensemble methods provide a highly robust baseline. They handle non-linear relationships effectively without the extensive hyperparameter tuning or scaling sensitivity inherent to support vector machines or neural networks.

### Evaluation Strategy
Because the provided Kaggle test set excludes target labels, the training data is split (80/20) to create a local validation set. This allows for rigorous calculation of precision, recall, and F1-scores before generating final predictions on the unlabelled test set.

---

## Impact

The current architecture achieves a 1.00 F1-score on the validation split. 

Beyond raw accuracy, the primary impact of this implementation is its interpretability. By extracting feature importances from the ensemble model, we can mathematically isolate the exact network behaviors that signal an intrusion. In this dataset, variables such as `src_bytes` and specific service rate differentials are heavily weighted by the model. For a security engineering team, this moves the system from a "black box" predictor to a diagnostic tool, allowing engineers to write highly targeted firewall rules or monitoring alerts based on the most critical features, rather than analyzing all network traffic uniformly.

---

## Further Steps

To advance this project from a static notebook to a production-ready system, consider the following engineering and mathematical improvements:

1. **Algorithm Benchmarking:** While Random Forest provides an excellent baseline, training and evaluating gradient boosting frameworks (like XGBoost or LightGBM) is recommended. These often yield faster inference times, which is a critical constraint when processing high-throughput network packets in real-time.
2. **Unsupervised Anomaly Detection:** The current supervised model requires labeled data and will only recognize known attack vectors. Implementing an unsupervised layer—such as an Isolation Forest or a localized Autoencoder—would allow the system to flag novel, zero-day network behaviors that deviate from the mathematical norm of standard traffic.
3. **Model Serialization:** Integrate `joblib` to export the fitted scaler and the trained model. This separates the training environment from the deployment environment, allowing the model to be loaded into a lightweight inference API.
4. **Cross-Validation:** Implement k-fold cross-validation to ensure the 1.00 F1-score is not a result of overfitting to a specific random split, guaranteeing the model's generalizability against unseen network signatures.
