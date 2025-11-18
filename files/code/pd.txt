


import pandas as pd

with open('data.csv', 'w') as f:
    f.write("Name,Age,City\n")
    f.write("Alice,30,New York\n")
    f.write("Bob,24,London\n")
    f.write("Charlie,35,Paris\n")

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}
df_to_save = pd.DataFrame(data)
df_to_save.to_csv('sample.csv', index=False)

# Read the CSV file, explicitly setting the header to the first row (index 0)
df = pd.read_csv('data.csv', header=0)
print(df)

df_dropped_rows = df.dropna()
df.dropna(how='all')
df.dropna(thresh=n)
df.fillna(value)
df.fillna(df.mean()) # median() or mode()

mean_A = df['A'].mean()
df['A'].fillna(mean_A, inplace=True)

sep: Specifies the delimiter used in the CSV file (e.g., sep='\t' for tab-separated values).
header: Specifies which row to use as the column names (e.g., header=None if there is no header, or header=2 to use the third row as the header).
names: Provides a list of column names if header=None or if you want to override the existing header.
index_col: Specifies which column(s) to use as the DataFrame index.
na_values: Specifies a list of strings to be recognized as NaN (Not a Number) values.
dtype: Specifies the data type for entire columns or specific columns.
nrows: Reads only a specified number of rows from the beginning of the file.
skiprows: Skips a specified number of rows at the beginning of the file.

df = pd.concat([df1, df2], ignore_index=True) # add rows from both data frames

# joins
df_outer = pd.merge(df1, df2, on=['id1', 'id2'], how='outer') # adds suffix _x and _y to recognize
df_inner = pd.merge(df1, df2, on=['id1', 'id2'], how='inner') 
df_left = pd.merge(df1, df2, on=['id1', 'id2'], how='left')

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
column_a = df['A']
print(column_a)

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
selected_columns = df[['A', 'C']]
print(selected_columns)

selected_with_loc = df.loc[:, ['A', 'C']] # Select all rows, and columns 'A' and 'C'
selected_with_iloc = df.iloc[:, [0, 2]] # Select all rows, and columns at index 0 and 2

filtered_df = df[(df['Age'] < 30) | (df['City'] == 'New York')] # or
filtered_df = df[(df['Age'] > 25) & (df['City'] == 'New York')] # and

df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

df['new_col_list'] = [7, 8, 9] # Assign a list of values
df['new_col_constant'] = 10 # Assign a single constant value (broadcasts to all rows)
df['col_sum'] = df['col1'] + df['col2'] # Assign the result of a calculation on existing columns

df['Total_Value'] = df.apply(lambda row: row['Quantity'] * row['Price'], axis=1)
df['Shipment_Size'] = df['Quantity'].apply(lambda x: 'Small' if x <= 3 else ('Medium' if x <= 10 else 'Large'))

normalized_df=(df-df.mean())/df.std()
normalized_df=(df-df.min())/(df.max()-df.min())

np_array = df.to_numpy() 
df = pd.DataFrame(numpy_array_2d, columns=['ColA', 'ColB', 'ColC'])
pd_series = pd.Series(numpy_array_1d)

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['x', 'y', 'z'])
s = pd.Series([7, 8, 9], index=['x', 'y', 'z'], name='C')
df['C'] = s

import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y)

# Set the x-axis label
plt.xlabel("X-axis values (units)")

# Set the y-axis label
plt.ylabel("Y-axis values (amplitude)")

# Add a title to the plot (optional)
plt.title("Sine Wave Plot with Axis Labels")

# Display the plot
plt.show()

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load a dataset (e.g., Iris dataset)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (species)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create a Logistic Regression model instance
# You can specify parameters like 'solver' and 'max_iter' if needed
model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=200)

# 4. Train the model using the training data
model.fit(X_train, y_train)

# 5. Make predictions on the test data
y_pred = model.predict(X_test)

# 6. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Optional: Predict probabilities
y_proba = model.predict_proba(X_test)
print(f"Predicted probabilities for the first sample: {y_proba[0]}")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Generate some sample data
# For simplicity, let's create a linear relationship with some noise
np.random.seed(0) # for reproducibility
X = 2 * np.random.rand(100, 1) # 100 samples, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + noise

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create a Linear Regression model
model = LinearRegression()

# 4. Train the model using the training data
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Model coefficients: {model.coef_[0][0]:.2f}")
print(f"Model intercept: {model.intercept_[0]:.2f}")

# 7. Visualize the results (optional)
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Prediction')
plt.legend()
plt.show()