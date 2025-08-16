Store Sales Forecasting Project
Overview
This project focuses on preprocessing sales data for multiple stores, creating time series sequences for LSTM-based forecasting, and visualizing daily sales per store. The dataset (store_sales.csv) contains sales data with features such as date, store ID, sales, promotions, and holidays. The project preprocesses the data, generates sequences for training machine learning models, and visualizes actual vs. predicted sales for each store using trained models.
Key tasks include:

Data Preprocessing: Loading, cleaning, and transforming sales data, including scaling, feature engineering (e.g., day of week, month, day of month), and one-hot encoding.
Time Series Sequence Creation: Generating sequences of historical data (e.g., 7, 14, or 30 days) for LSTM input, split into training, validation, and test sets.
Model Evaluation and Visualization: Evaluating trained LSTM models for each store and plotting actual vs. predicted sales on a daily basis.

The project is implemented in Python using libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and TensorFlow.
Project Structure

store_sales.csv: Input dataset containing sales data for multiple stores.
Code_file.ipynb: Jupyter notebook with the complete code for data preprocessing, sequence creation, model evaluation, and visualization.
scalers.pkl: Saved MinMaxScaler object for inverse transforming sales predictions.
best_model_store_X.joblib: Trained LSTM models for each store (X ranges from 1 to 10).
README.md: This file, providing an overview and instructions for the project.

Setup Instructions
To run this project locally, follow these steps:
Prerequisites

Python 3.12.4 or higher
Jupyter Notebook or JupyterLab
Required Python libraries:pip install pandas numpy scikit-learn matplotlib tensorflow joblib



Installation

Clone the repository:git clone https://github.com/your-username/store-sales-forecasting.git
cd store-sales-forecasting


Install the required dependencies:pip install -r requirements.txt

(Create a requirements.txt file with the above libraries if not already present.)
Ensure the store_sales.csv file and saved models (scalers.pkl, best_model_store_X.joblib) are in the project directory.

Running the Notebook

Launch Jupyter Notebook:jupyter notebook


Open Code_file.ipynb in the Jupyter interface.
Run the cells sequentially to preprocess the data, recreate test sequences, evaluate models, and generate visualizations.

Usage
The notebook is divided into sections:

Data Preprocessing:

Loads store_sales.csv and converts dates to datetime format.
Scales sales data using MinMaxScaler for each store.
Extracts features like day of week, month, and day of month, with one-hot encoding for categorical variables.
Creates time series sequences (7, 14, or 30 days) for LSTM input.


Data Splitting:

Splits data into training (70%), validation (15%), and test (15%) sets for each store.
Ensures chronological order by disabling shuffling in train_test_split.


Model Evaluation and Visualization:

Loads pre-trained LSTM models and the scaler.
Evaluates models on the test set using metrics like MAE, RMSE, and MAPE.
Plots actual vs. predicted sales for each store on a daily basis.



To generate visualizations for a specific store:

Run the second code cell in the notebook, which iterates over all stores (1 to 10) and produces plots.
Modify the stores list in the notebook to focus on specific stores if needed.

Data Description
The store_sales.csv dataset includes:

date: Date of the sales record (converted to datetime).
store: Store ID (1 to 10).
sales: Daily sales amount (float).
promo: Binary indicator for promotional activities (0 or 1).
holiday: Binary indicator for holidays (0 or 1).

Derived features:

day_of_week: Day of the week (0=Monday, 6=Sunday).
month_scaled: Month scaled to [0, 1].
day_of_month_scaled: Day of the month scaled to [0, 1].
dow_X: One-hot encoded columns for day of week (X from 0 to 6).
lagged_sales: Sales lagged by 7 days.
moving_avg_7: 7-day moving average of scaled sales.

Model Details

Model Type: LSTM (Long Short-Term Memory) neural network.
Hyperparameters: Vary by store, with sequence lengths (7, 14, or 30 days), units (50 or 100), and learning rates (0.001 or 0.01) optimized per store.
Metrics:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Mean Absolute Percentage Error (MAPE)



Visualizations
For each store, the notebook generates a plot comparing actual vs. predicted sales on the test set. The x-axis represents dates, and the y-axis represents sales (inverse-transformed to the original scale). Blue lines indicate actual sales, and red lines indicate predicted sales.
Notes

The dataset is assumed to have no missing values, as per the preprocessing step.
Models and scalers must be present in the project directory (scalers.pkl and best_model_store_X.joblib).
The sequence length for each store is predefined based on prior hyperparameter tuning results.

Future Improvements

Incorporate additional features (e.g., external economic indicators).
Experiment with other time series models (e.g., Transformer-based models).
Automate hyperparameter tuning using grid search or Bayesian optimization.
Add interactive visualizations using Plotly or Bokeh.

Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bug fixes.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, please contact your-email@example.com or open an issue on GitHub.
