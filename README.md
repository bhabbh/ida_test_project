
# Store Inventory Forecasting Application

This Python project helps store owners determine how many units of a given product they should order for the upcoming n days. The application is designed to provide accurate predictions based on historical sales data and user inputs.

## Prerequisites

Before running the application, make sure you have the following dependencies installed:

1.  **Conda**: You need to have Conda installed. If you don't have it, you can download and install it from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

## Setup

Follow these steps to set up the environment and run the application:
    
-   Navigate to the project's root directory in your terminal
    
	  `cd project_ida_test` 
    
-   Create a new Conda environment named "ida-test" by running the following commands:
    	   
	   `conda create --name ida_test python=3.11` 
    
	   `conda activate ida_test` 
	   
	   `conda install pip` 
    
-   Install the required Python packages by running:    

	 `pip install -r requirements.txt` 
    

## Running the Application

Once you have set up the environment, you can run the application. Follow these steps:

1.  Whenever you receive new data, start with training your model by running the folliwing command from the root directory of the project:
	`python -m run train` 

2. Start the backend server by running the following command from the root directory of the project:    

	`python -m run back` 
    
    Wait until the Application startup is complete, and Uvicorn is running on [http://0.0.0.0:8000](http://0.0.0.0:8000).
    
	Open a new terminal window (while keeping the backend running) and run the front-end of the application:    

	 `python -m run front` 
    
    This will launch a web browser, displaying the application's user interface.
    
3.  In the web browser, select a store from the drop-down menu, specify a prediction date, set the range of days, and enter a list of sale IDs.
    
4.  Click the "Predict" button to generate the product order recommendations based on your inputs.
    

## Usage

-   Select Store: Choose the store for which you want to make inventory predictions.
	ex : "c8c368f0311ea25b581cb3c704fe3a70"
-   Prediction Date: Specify the date for which you want to predict product demand.
	ex : 2023/12/10
-   Sale IDs: Enter a list of sale IDs relevant to the selected store.
	ex : "5097825564f0cfe61b9b544661ad0ae0, fffe30ef33248001900c5c5bd627562f, 097b7da23b3925f456505483315860a5"

## Test

From the root directory, run :

	`pytest` 

