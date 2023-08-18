# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn for styling
sns.set(style='darkgrid')  # Set Seaborn's style

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Reading data from file
df = pd.read_csv('RELIANCE.NS.csv')

# Data preparations
df.index = pd.to_datetime(df['Date'])
df.drop(['Date'], axis='columns', inplace=True)

# Create predictor variables
df['Open-Close'] = df['Open'] - df['Close']
df['High-Low'] = df['High'] - df['Low']

# Store all predictor variables in a variable X
X = df[['Open-Close', 'High-Low']]

# Target variables
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Now split the data into train and test
split_percentage = 0.8
split = int(split_percentage * len(df))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]

# Support vector classifier
cls = SVC().fit(X_train, y_train)
df['Predicted_Signal'] = cls.predict(X)

# Calculate daily returns
df['Return'] = df['Close'].pct_change()

# Calculate strategy returns
df['Strategy_Return'] = df['Return'] * df['Predicted_Signal'].shift(1)

# Calculate Cumulative returns
df['Cum_Ret'] = df['Return'].cumsum()

# Plot Strategy Cumulative returns
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(df['Cum_Ret'], color='red', label='Cumulative Return')
plt.plot(df['Cum_Strategy'], color='blue', label='Cumulative Strategy Return')
plt.title('Cumulative Returns vs. Cumulative Strategy Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
