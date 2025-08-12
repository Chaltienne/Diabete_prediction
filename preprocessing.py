import pandas as pd
import pickle
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_preprocess_and_train_model():
    try:
        # Load data
        print("Downloading dataset...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                   'DiabetesPedigreeFunction', 'Age', 'Outcome']
        data = pd.read_csv(url, names=columns)
        
        if data.empty:
            raise ValueError("Downloaded dataset is empty.")
        
        # Preprocess: Replace zero values with mean for specific columns
        print("Preprocessing data...")
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            data[col] = data[col].replace(0, data[col].mean())
        
        # 🔹 Sauvegarder le jeu de données prétraité
        with open('diabetes_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        # Split features and target
        print("Splitting data...")
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Save model and scaler
        print("Saving model and scaler...")
        with open('diabetes_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        return accuracy
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        raise
    except Exception as e:
        print(f"Error in processing or training: {e}")
        raise

if __name__ == "__main__":
    accuracy = load_preprocess_and_train_model()
    print(f"Model trained with accuracy: {accuracy:.2%}")
