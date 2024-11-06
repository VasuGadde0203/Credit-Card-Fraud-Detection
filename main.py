import preprocessing as pp
import pandas as pd
import modelling as md
from sklearn.model_selection import train_test_split

# Load and preprocess data
file_path = r'C:\\Users\\vasug\\OneDrive\\Desktop\\machine learning\\projects\\credit card fraud detection\\creditcard.csv'
data = pp.load_data(file_path)
df = pd.read_csv(file_path)
X, Y = pp.preprocess_data(df)
X = pp.scale_features(X)
X_pca = pp.apply_pca(X)


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.2, stratify=Y, random_state=2)

# Train and evaluate the model
model = md.train_model(X_train, Y_train)
train_accuracy, test_accuracy, report = md.evaluate_model(model, X_train, Y_train, X_test, Y_test)

# Print results
print(f'Accuracy on Training data: {train_accuracy}')
print(f'Accuracy score on Test Data: {test_accuracy}')
print("\nClassification Report:\n", report)
