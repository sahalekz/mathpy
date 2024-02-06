from google.colab import files
import pandas as pd
import matplotlib.pyplot as plt


# Upload the dataset file
uploaded = files.upload()



df = pd.read_csv('DATA.csv')  # Change 'your_uploaded_file.csv' to your actual file name

# Compute Mean
mean_value = df.mean()
print("Mean:", mean_value)

# Compute Median
median_value = df.median()
print("Median:", median_value)

# Compute Mode
mode_value = df.mode().iloc[0]
print("Mode:", mode_value)

# Standard Deviation
std_deviation = df.std()
print("\nStandard Deviation:\n", std_deviation)

# Assuming you want to find the correlation between all columns
correlation_matrix = df.corr()

print("Correlation Matrix:\n", correlation_matrix)

from sklearn.linear_model import LinearRegression

x = df[['AGE']]
y = df['PART']

model = LinearRegression()
model.fit(x, y)

# Coefficients
slope = model.coef_[0]
intercept = model.intercept_

print("Linear Regression Coefficients:")
print("Slope:", slope)
print("Intercept:", intercept)
plt.plot(x,y_pred,color='red',linewidth=3)