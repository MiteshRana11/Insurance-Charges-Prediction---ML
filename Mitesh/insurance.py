import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

df = pd.read_csv('insurance.csv')

#Top 5 values of the dataset
df.head()

#Number of columns and rows of Dataset
df.shape

# Display information about the DataFrame, including column names, data types, and memory usage
df.info()

#Converting bmi column into int
df['bmi'] = df['bmi'].astype(int)

df.info()

#Checking for the null values
df.isnull().sum()

#Checking for unique values in region column
df['region'].unique()

#Checking for unique values in smoker column
df['smoker'].unique()

import missingno as msno 
msno.matrix(df)

#Creating a bar graph of Sex vs Charges
plt.bar(df['sex'],df['charges'])
plt.title('Sex vs Charges')
plt.xlabel('Sex')
plt.ylabel('Charges')

# Calculate the total charges for each sex
charges_by_sex = df.groupby('sex')['charges'].sum()

# Create the pie chart
plt.pie(charges_by_sex, labels=charges_by_sex.index, autopct='%1.1f%%')
plt.title('Sex vs Charges')

#Creating a bar graph of age vs Charges
plt.bar(df['age'],df['charges'])
plt.title('Age vs Charges')
plt.xlabel('Age')
plt.ylabel('Charges')

# Calculate the total charges for each region
charges_by_region = df.groupby('region')['charges'].sum()

# Create the pie chart
plt.pie(charges_by_region, labels=charges_by_region.index, autopct='%1.1f%%')
plt.title('Region vs Charges')

#Creating a bar graph of bmi vs Charges
plt.scatter(df['bmi'],df['charges'])
plt.title('bmi vs Charges')
plt.xlabel('bmi')
plt.ylabel('Charges')

#Importing LabelEncoder & OneHotEncoder from sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Creating Objects of LabelEncoder & OneHotEncoder
Le = LabelEncoder()
OHE = OneHotEncoder(sparse=False)

df['smoker'] = Le.fit_transform(df['smoker'])

df['sex'] = Le.fit_transform(df['sex'])

df

x=df['region'].values.reshape(-1,1)


OHE = OneHotEncoder(sparse=False)  # Set sparse to False

# Fit and transform the encoded features
encoded_features = OHE.fit_transform(x)

# Check the type of the encoded features
if isinstance(encoded_features, np.ndarray):
    # Handle NumPy array
    new_columns = OHE.get_feature_names_out(['feature_name'])
elif isinstance(encoded_features, scipy.sparse.csr_matrix):
    # Handle sparse matrix
    new_columns = OHE.get_feature_names_out(['feature_name']).tolist()
else:
    raise ValueError("Unexpected type for encoded features")

print(new_columns)


df_encoded = pd.DataFrame(encoded_features, columns=new_columns) 


df = pd.concat([df,df_encoded],axis=1)

df.drop(columns='region',axis=1,inplace=True)

df.head()

#Checking the name of the total columns present in df
df.columns

from sklearn.preprocessing import RobustScaler
import numpy as np

# Assuming 'data' is the array-like object containing your data

# Create the RobustScaler object
scaler = RobustScaler()

# Fit the scaler to your data
scaler.fit(df[['charges']])

# Transform the data using the scaler
df['charges'] = scaler.transform(df[['charges']])

# Alternatively, you can fit and transform in one step
df['charges'] = scaler.fit_transform(df[['charges']])

# Print the scaled data
print(df['charges'])

# Extract features (X) and target variable (y)
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'feature_name_southwest', 'feature_name_southeast', 'feature_name_northwest']]
y = df['charges']

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

print('X_train_shape: ',X_train.shape)
print('y_train_shape: ',y_train.shape)
print('X_test_shape: ',X_test.shape)
print('y_test_shape: ',y_test.shape)

from sklearn.preprocessing import StandardScaler

SD = StandardScaler()

X_train_SD = SD.fit_transform(X_train)

X_test_SD = SD.transform(X_test)

X_mean = X_train_SD.mean(axis=0)

y_mean = y_train.mean(axis=0)

num = 0
dim = 0
epsilon = 1e-8  
for i in range(len(X_train_SD)):
    num += (X_train_SD[i] - X_mean) * (y_train[i] - y_mean)
    dim += (X_train_SD[i] - X_mean) ** 2

coff = num / (dim + epsilon)
inter = y_mean - (coff * X_mean)
print('Coff:', coff)
print('Intercept:', inter)

m = coff
c = inter
y = m * 11 + c
z = m * 95644.50 + c
print('y:', y)
print('z:', z)


y_pred = X_test_SD.dot(coff)
mse = np.mean((y_pred - y_test) ** 2)
mae = np.mean(np.abs(y_pred - y_test))
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

print('Mean Squared Error (MSE):', mse)
print('Mean Absolute Error (MAE):', mae)
print('R-squared:', r2)


plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Linear Regression - Actual vs. Predicted Charges')
plt.show()

