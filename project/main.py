import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('Real estate.csv')
# Remove any rows with missing values
data.dropna(inplace=True)
# Remove any duplicate rows
data.drop_duplicates(inplace=True)
# Select the relevant columns
cols = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station'
    , 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
print(data.head())
print(data.info())
print(data.describe())
print(data.columns) #shows the columns

# Select the relevant columns
X = data[cols].values
y = data['Y house price of unit area'].values

# showing different relations with x columns and y before regression
fig, axs = plt.subplots(1, X.shape[1], figsize=(14, 6))
fig.subplots_adjust(wspace=1)
for i in range(X.shape[1]):
    axs[i].scatter(X[:, i], y, alpha=0.5)
    axs[i].set_xlabel(data.columns.array[i + 1], fontsize=7)
    axs[i].set_ylabel('House Price')
    axs[i].set_title("corr:" + str(data[cols[i]].corr(data['Y house price of unit area']))[:8])
plt.show()

# correlation between columns before regression
fig, axs = plt.subplots(6, 6, figsize=(15, 15))
fig.subplots_adjust(hspace=1)
fig.subplots_adjust(wspace=1)
for i in range(6):
    colCount = 0
    for j in range(6):
        x_col1 = cols[i]
        x_col2 = cols[j]
        axs[i, colCount].scatter(data[x_col1], data[x_col2], alpha=0.5)
        axs[i, colCount].set_xlabel(x_col1, fontsize=6)
        axs[i, colCount].set_ylabel(x_col2, fontsize=6)
        axs[i, colCount].set_title("corr:" + str(data[x_col1].corr(data[x_col2]))[:9], fontsize=7)
        colCount += 1
plt.show()

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model on the training data
reg = LinearRegression().fit(X_train, y_train)

# Make predictions on the testing data
y_pred = reg.predict(X_test)

# Calculate the mean squared error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot the actual vs predicted values in different colors
colors = y_test / np.max(y_test)  # normalize the actual house prices to [0, 1]
plt.scatter(y_test, y_pred, c=colors, cmap='coolwarm')
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.title('Linear Regression Model')
plt.show()


# Print the results
print("Coefficients: ", reg.coef_)
print("Intercept: ", reg.intercept_)
print("Mean squared error: {:.2f}".format(mse))
print("R-squared score: {:.2f}".format(r2))

