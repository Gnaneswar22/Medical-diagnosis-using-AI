import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle

def train_diabetes_model():
    try:
        # Load the diabetes dataset
        data = pd.read_csv('Datasets/diabetes_data.csv')
        
        # Separate features and target
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model with probability support
        model = SVC(probability=True, kernel='rbf', random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save the model
        with open('Models/diabetes_model.sav', 'wb') as f:
            pickle.dump(model, f)
        
        # Calculate and print accuracy
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)
        
        print(f"Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"Testing Accuracy: {test_accuracy*100:.2f}%")
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error training model: {e}")

if __name__ == "__main__":
    train_diabetes_model()
