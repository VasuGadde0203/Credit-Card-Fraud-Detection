# Credit Card Fraud Detection
This repository contains a machine learning project for detecting fraudulent credit card transactions using a highly imbalanced dataset. The model utilizes feature engineering, dimensionality reduction, and advanced sampling techniques to improve accuracy and reliability.

## Project Overview
Credit card fraud is a serious issue in today's digital economy, where unauthorized transactions can lead to significant financial loss. This project aims to detect fraudulent transactions using machine learning techniques, ensuring accurate results on a highly imbalanced dataset.

## Key Features
- **Data Preprocessing:** Cleaning and preparing data, with emphasis on handling class imbalance.
- **Feature Engineering:** Scaling the transaction amount, dimensionality reduction with PCA, and time-based feature extraction.
- **Sampling Techniques:** Balancing classes using SMOTE to prevent model bias toward non-fraudulent transactions.
- **Model Training:** Implemented a Logistic Regression model with tuned parameters.
- **Performance Metrics:** High accuracy on both training and testing data, demonstrating model effectiveness.

## Dataset 
The dataset used in this project is from Kaggle, containing over 284,000 transactions. It includes only anonymized numerical features (V1-V28) and two additional features:
  - **Amount:** Transaction amount.
  - **Class:** Indicates whether the transaction is fraudulent (1) or legitimate (0).
**Note:** The dataset is highly imbalanced, with only 492 fraudulent transactions out of 284,807 (approximately 0.17%).

## Model Performance
The model's performance on the dataset is summarized below:
- **Training Accuracy:** 98.82%
- **Testing Accuracy:** 97.64%
These metrics reflect the model's ability to generalize well and detect fraudulent transactions effectively.

## Project Structure
The code is split into separate files for modularity and ease of understanding:
- **preprocessing.py:** Contains data preprocessing steps, including scaling and PCA.
- **sampling.py:** Contains the SMOTE sampling technique for class balancing.
- **modeling.py:** Includes model training and evaluation code.
- **main.py:** Orchestrates the flow by calling the preprocessing, sampling, and modeling modules.

## Requirements
The project requires the following Python libraries:
- numpy
- pandas
- scikit-learn

You can install the required libraries using:
- pip install -r requirements.txt

## Usage
- To run the project locally:

- **Clone the repository:**
  - git clone https://github.com/VasuGadde0203/Credit-Card-Fraud-Detection.git
  - cd Credit-Card-Fraud-Detection

- Execute main.py to run the preprocessing, model training, and evaluation.

## Results
- The model achieves excellent accuracy and robustness with the following metrics:
- **Metric	Score**
  - Training Accuracy	98.82%
  - Testing Accuracy	97.64%

## Future Improvements
  - **Feature Engineering:** More complex features, such as user-based transaction histories, could improve accuracy.
  - **Hyperparameter Tuning:** Fine-tuning model parameters to further enhance performance.
- **Model Ensembles:** Using ensemble methods such as Random Forest or XGBoost for potentially higher accuracy.

## License
This project is licensed under the MIT License.

## Acknowledgements
Kaggle for providing the dataset.

