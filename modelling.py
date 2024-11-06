from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, Y_train):
    # Initialize and train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    # Evaluate on training data
    X_train_prediction = model.predict(X_train)
    train_accuracy = accuracy_score(Y_train, X_train_prediction)

    # Evaluate on test data
    X_test_prediction = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, X_test_prediction)

    # Detailed evaluation report
    report = classification_report(Y_test, X_test_prediction)
    return train_accuracy, test_accuracy, report
