1. **Create a New GitHub Repository**:
   - Go to your GitHub account and create a new repository named something like `network-capacity-planning`.

2. **Clone the Repository**:
   - Once the repo is created, clone it locally:
     ```bash
     git clone https://github.com/<your-username>/network-capacity-planning.git
     ```

3. **Set Up Directory Structure**:
   Organize the repository for clarity:
   ```
   network-capacity-planning/
   ├── data/               # Raw and processed datasets
   ├── notebooks/          # Jupyter notebooks for analysis and model training
   ├── scripts/            # Python scripts for data processing and model training
   ├── models/             # Saved ML models
   ├── visuals/            # Matplotlib plots and visualizations
   ├── requirements.txt    # Python dependencies
   ├── README.md           # Project overview and instructions
   └── LICENSE             # License file
   ```

4. **Add the Python Environment Files**:
   Create a `requirements.txt` to manage dependencies:
   ```txt
   pandas
   numpy
   matplotlib
   scikit-learn
   jupyterlab
   ```

5. **Push to GitHub**:
   After adding the files:
   ```bash
   git add .
   git commit -m "Initial commit for network capacity planning use case"
   git push origin master
   ```

---

### Code for Network Capacity Planning (Use Case 8)

#### Step 1: Data Collection and Preprocessing
We assume you have historical network traffic data with features like `timestamp`, `bandwidth_utilization`, `num_users`, and `latency`.

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('data/network_traffic.csv')

# Data preview
print(df.head())

# Feature engineering: Create time-based features if needed
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Drop timestamp column for now
df = df.drop(['timestamp'], axis=1)

# Split data into features (X) and target (y)
X = df.drop('bandwidth_utilization', axis=1)
y = df['bandwidth_utilization']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Step 2: Model Building
We will use **Linear Regression** to predict future network capacity.

```python
# Build the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted Bandwidth Utilization')
plt.legend()
plt.show()
```

#### Step 3: Saving the Model
You can save the trained model for future use.

```python
import pickle

# Save the model
with open('models/network_capacity_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

#### Step 4: Load and Use the Model for Predictions
To use the saved model for future predictions:

```python
# Load the saved model
with open('models/network_capacity_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Predict with new data
new_data = np.array([[15, 3]])  # Example input (hour=15, day_of_week=3)
predicted_bandwidth = loaded_model.predict(new_data)
print(f'Predicted Bandwidth Utilization: {predicted_bandwidth}')
```

---

### `README.md` for GitHub Repository

```md
# Network Capacity Planning Using AI/ML

## Project Overview

This project demonstrates how to use AI and Machine Learning to forecast future network capacity needs based on historical traffic data. By leveraging Python libraries such as **Pandas**, **Numpy**, **Matplotlib**, and **Scikit-learn**, we predict bandwidth utilization and help in capacity planning for network infrastructure.

## Use Case: Network Capacity Planning

### Problem
Network administrators often need to forecast network capacity to handle future traffic loads and prevent congestion. This project uses historical data on network traffic to predict bandwidth utilization in the future, allowing proactive planning for infrastructure upgrades.

### Approach
We utilize the following steps:
1. **Data Collection**: Historical network traffic data, including features such as bandwidth utilization, number of users, and network latency.
2. **Data Preprocessing**: Clean and transform the data for analysis. Time-based features such as hour and day of the week are added.
3. **Model Building**: A **Linear Regression** model is trained to predict future bandwidth needs.
4. **Evaluation**: The model is evaluated using **Mean Squared Error (MSE)** and predictions are compared with actual usage data.
5. **Visualization**: Data visualizations using **Matplotlib** show trends and performance.

### Project Structure
```bash
network-capacity-planning/
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter Notebooks for analysis and model training
├── scripts/            # Python scripts for data processing and model training
├── models/             # Saved ML models
├── visuals/            # Plots and visualizations
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

### Getting Started

#### 1. Clone the Repository
```bash
git clone https://github.com/umerjillani/network-capacity-planning.git
cd network-capacity-planning
```

#### 2. Install Dependencies
Install the necessary Python packages via pip:
```bash
pip install -r requirements.txt
```

Or, create a Conda environment:
```bash
conda create --name network-automation python=3.8
conda activate network-automation
conda install --file requirements.txt
```

#### 3. Run Jupyter Notebook
To explore the data and the model, open Jupyter Lab:
```bash
jupyter lab
```

#### 4. Run the Python Scripts
You can execute the scripts directly to preprocess the data, train the model, and make predictions:
```bash
python scripts/train_model.py
```

### Visualizations
Key visualizations generated in this project are available in the `visuals/` folder, showcasing trends in bandwidth utilization and model performance.

### Model
The trained Linear Regression model is stored in the `models/` directory. You can load it to make predictions on new data.

### Future Enhancements
- Incorporate real-time traffic analysis for dynamic capacity planning.
- Explore more advanced models like Time-Series Forecasting (ARIMA, LSTM).
- Integrate with network automation tools to dynamically adjust network configurations based on predictions.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

### Push the Repository to GitHub
Once you've completed the steps and added the code and `README.md`, push everything to your GitHub repo:

```bash
git add .
git commit -m "Added code and README for network capacity planning use case"
git push origin master
```

This setup will give a detailed and professional presentation of your work for **Use Case 8: Network Capacity Planning**.
