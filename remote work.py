import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
# Update the file path to the correct location on your local system
file_path = 'Impact_of_Remote_Work_on_Mental_Health..csv'
data = pd.read_csv(file_path)

# Continue with the rest of the data processing


# Step 1: Statistical Summary
print("Statistical Summary of Numerical Features:")
print(data.describe())

# Step 2: Exploratory Data Analysis (EDA)

# Histogram for Age distribution
plt.figure(figsize=(10, 5))
plt.hist(data['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution of Employees')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Bar plot for Work Location distribution
plt.figure(figsize=(10, 5))
data['Work_Location'].value_counts().plot(kind='bar', color='coral', edgecolor='black')
plt.title('Work Location Distribution')
plt.xlabel('Work Location')
plt.ylabel('Number of Employees')
plt.show()

# Scatter plot for Hours Worked vs. Work-Life Balance Rating
plt.figure(figsize=(10, 5))
plt.scatter(data['Hours_Worked_Per_Week'], data['Work_Life_Balance_Rating'], alpha=0.6, c='green')
plt.title('Hours Worked vs. Work-Life Balance Rating')
plt.xlabel('Hours Worked Per Week')
plt.ylabel('Work-Life Balance Rating')
plt.show()

# Step 3: Correlation Analysis
# Select only the numerical columns for correlation analysis
numerical_data = data.select_dtypes(include=[np.number])

# Calculate the correlation matrix for numerical features
correlation_matrix = numerical_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Step 4: Insights Extraction

# Analysis of mental health conditions based on work location
mental_health_by_location = data.groupby('Work_Location')['Mental_Health_Condition'].value_counts(normalize=True).unstack()
print("\nMental Health Condition Distribution by Work Location:")
print(mental_health_by_location)

# Analysis of Work-Life Balance Rating based on Job Role
avg_balance_by_role = data.groupby('Job_Role')['Work_Life_Balance_Rating'].mean().sort_values()
print("\nAverage Work-Life Balance Rating by Job Role:")
print(avg_balance_by_role)

# Displaying the plots and summaries for analysis
plt.figure(figsize=(10, 5))
avg_balance_by_role.plot(kind='bar', color='purple', edgecolor='black')
plt.title('Average Work-Life Balance Rating by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Average Work-Life Balance Rating')
plt.show()
