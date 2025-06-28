import os
import pickle

def check_models():
    model_dir = "Models"
    model_files = [
        'diabetes_model.sav',
        'heart_disease_model.sav',
        'parkinsons_model.sav',
        'lungs_disease_model.sav',
        'Thyroid_model.sav'
    ]

    print("Checking model files...")
    for model_file in model_files:
        file_path = os.path.join(model_dir, model_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                    print(f"✓ {model_file}: Successfully loaded")
            except Exception as e:
                print(f"✗ {model_file}: Error loading - {str(e)}")
        else:
            print(f"✗ {model_file}: File not found")

if __name__ == "__main__":
    check_models()
