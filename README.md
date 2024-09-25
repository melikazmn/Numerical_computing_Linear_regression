# Real Estate Price Prediction with Linear Regression

## Overview

This project implements a linear regression model to predict house prices based on various features from a real estate dataset. The dataset contains information such as transaction dates, house age, distance to the nearest MRT station, the number of convenience stores nearby, and geographical coordinates (latitude and longitude). The aim is to explore relationships among these variables and develop a predictive model for house prices.

## Features

- Data cleaning: Handles missing values and duplicates.
- Exploratory Data Analysis (EDA):
  - Visualizes relationships between features and house prices.
  - Correlation analysis among features.
- Implements linear regression using `scikit-learn`.
- Evaluates the model using Mean Squared Error (MSE) and R-squared metrics.
- Visualizes the actual vs. predicted house prices.

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

## Dataset

The dataset used for this project is a CSV file named Real estate.csv. It contains the following columns:

- X1 transaction date: Date of the transaction.
- X2 house age: Age of the house in years.
- X3 distance to the nearest MRT station: Distance in meters to the nearest MRT station.
- X4 number of convenience stores: Count of convenience stores nearby.
- X5 latitude: Latitude of the house location.
- X6 longitude: Longitude of the house location.
- Y house price of unit area: Target variable representing the price per unit area of the house.

## Results

The project produces various visualizations to aid in understanding the data and the effectiveness of the linear regression model:

Scatter plots showing relationships between individual features and house prices.
A correlation matrix visualizing relationships among all features.
A scatter plot comparing actual house prices against predicted prices.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please fork the repository and submit a pull request.
