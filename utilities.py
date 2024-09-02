import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")


def load_excel_sheets(file_path):
    '''Load all sheets from an Excel file and return a dictionary of DataFrames.'''
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    dfs = {}
    for sheet_name in sheet_names:
        try:
            dfs[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"DataFrame loaded successfully for sheet: {sheet_name}")
        except Exception as e:
            print(f"Error loading DataFrame for sheet: {sheet_name}")
            print(e)
    return dfs


def drop_unnamed_columns(df):
    '''Drop columns with names containing 'Unnamed' from a DataFrame.'''
    # Filter columns containing 'Unnamed' in their names
    unnamed_columns = [col for col in df.columns if 'Unnamed' in col]

    # Drop the identified columns
    df.drop(columns=unnamed_columns, inplace=True)

    # Return the DataFrame with unnamed columns dropped
    return df

def missing_value_percent(df):
    '''Calculate the percentage of missing values in each column of a DataFrame.'''
    # Calculate the total number of missing values in each column
    missing_values = df.isna().sum()

    # Calculate the total number of values in each column
    total_values = df.shape[0]

    # Calculate the percentage of missing values for each column
    missing_percent = (missing_values / total_values) * 100

    # Create a DataFrame to store the results
    missing_df = pd.DataFrame({
        'Column': missing_percent.index,
        'Missing_Percent': missing_percent.values
    })

    return missing_df

def drop_high_missing_columns(df, threshold):
    '''Drop columns with more than a specified percentage of missing values from a DataFrame.'''
    # Calculate the percentage of missing values for each column
    missing_percent = (df.isna().sum() / df.shape[0]) * 100

    # Filter columns with missing value percentage greater than the threshold
    high_missing_columns = missing_percent[missing_percent > threshold].index

    # Drop the identified columns from the DataFrame
    df.drop(columns=high_missing_columns, inplace=True)

    return df

def violin_plot_all_columns(df):
    '''Create a violin plot for each column in a DataFrame.'''
    # Set the style of the plot
    sns.set(style="whitegrid")

    # Get the list of columns
    columns = df.columns

    # Set the size of the plot
    plt.figure(figsize=(20, 10))

    # Iterate over each column and create a violin plot
    for i, column in enumerate(columns, 1):
        plt.subplot(1, len(columns), i)
        sns.violinplot(y=df[column], color='skyblue')
        plt.title(f'Violin Plot for {column}')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    '''Calculate the correlation matrix and plot it as a heatmap.'''
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Correlation Matrix')
    plt.show()

def plot_feature_importances_regression(X, y):
    '''Train regression models and plot feature importances.'''
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
    }
    
    # Initialize dictionary to store feature importances
    importances = {}
    
    # Train models and get feature importances
    for name, model in models.items():
        model.fit(X_train, y_train)
        if hasattr(model, 'feature_importances_'):
            importances[name] = dict(sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1]))
        elif hasattr(model, 'coef_'):
            importances[name] = dict(sorted(zip(X.columns, np.abs(model.coef_)), key=lambda x: x[1]))

    # Plot feature importances for each model
    plt.figure(figsize=(12, 6 * len(models)))
    for i, (name, importance) in enumerate(importances.items(), start=1):
        plt.subplot(len(models), 1, i)
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))  # Generate colors
        features, scores = zip(*importance.items())
        plt.barh(features, scores, color=colors)  # Horizontal bar chart
        plt.title(f'{name} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')

    plt.tight_layout()
    plt.show()

def evaluate_regression_model(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train the regression model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Plot true vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predictions')  # Predicted values in blue
    plt.scatter(y_test, y_test, color='red', label='True Values')  # True values in red
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("TOC prediction for Kolmani river 2")
    plt.grid(True)
    plt.legend()

    # Display evaluation metrics on the chart
    metrics_text = f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR^2: {r_squared:.2f}"
    plt.text(0.1, 0.6, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.show()



def perform_feature_selection(X, y):
    # Record the start time
    start_time = time.time()

    cv = StratifiedKFold(5)  # For regression tasks, you may want to use regular KFold
    visualizer = RFECV(RandomForestRegressor(), cv=cv, scoring='neg_mean_squared_error')

    visualizer.fit(X, y)  # Fit the data to the visualizer
    visualizer.show()

    # Access the selected features
    selected_features = visualizer.support_

    # Print the indices of selected features
    print("Selected Features Indices:", [i for i, selected in enumerate(selected_features) if selected])

    # Record the end time
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time

    # Print the time taken
    print(f"Time taken: {time_taken} seconds")

