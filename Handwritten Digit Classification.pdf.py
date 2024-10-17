import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to load data from a CSV file
def load_data(filepath):
    # Read data from CSV file, ensuring no headers are interpreted
    data = pd.read_csv(filepath, header=None)
    # Features are all columns after the first
    X = data.iloc[:, 1:].values
    # Labels are in the first column
    y = data.iloc[:, 0].values
    return X, y

def main():
    # Filepaths to the training and testing dataset CSV files
    train_path = 'D:/Master Information Technolgies/Subjects/ML/Data_Set/mnist_train.csv'
    test_path = 'D:/Master Information Technolgies/Subjects/ML/Data_Set/mnist_test.csv'

    # Load the training and testing data
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    # Initialize the Gaussian Naive Bayes classifier
    gnb = GaussianNB()

    # Fit the model on the training data
    gnb.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = gnb.predict(X_test)

    # Calculate and print the accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
