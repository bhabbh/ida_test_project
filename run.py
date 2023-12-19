"""Main run script"""

import argparse
import subprocess
import os

root_dir = os.path.abspath(os.getcwd())
api_dir = os.path.join(root_dir,"ida_test/api")
app_front_path = os.path.join(api_dir,"app_front.py")
app_back_path = os.path.join(api_dir,"app_back.py")

def train_model():
    # Train model and save it
    subprocess.run(["python", "-m", "ida_test.model_scripts.train_save_model"])

def run_back():
    # Start the back-end (FastAPI) part
    subprocess.run(["uvicorn", "ida_test.api.app_back:app", "--host", "0.0.0.0", "--port", "8000"])

def run_front():
    # Start the front-end (Streamlit) part
    subprocess.run(["streamlit", "run", app_front_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the front-end or back-end part of your application.")
    parser.add_argument("part", choices=["train", "back", "front"], help="Specify 'train' to train the model, 'back' to run the back-end or 'front' to run the front-end.")

    args = parser.parse_args()

    if args.part == "back":
        run_back()
    elif args.part == "front":
        run_front()
    elif args.part == "train":
        train_model()