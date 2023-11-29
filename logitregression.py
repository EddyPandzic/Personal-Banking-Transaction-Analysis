# logitregression.py
# Author: Eddy Pandzic
# Transaction Probability Prediction with Logistic Regression model


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression


df = pd.read_excel("2019-2022_Transaction_DataT.xlsx")

df["Amount_abs"] = abs(df.Amount)
df = df.drop(["Description", "Location", "Amount", "Miscellaneous"], axis=1)
print(df)

# Covert excel's True and False values for binary 0 and 1 for use in statistical inferences and mathematical purposes
# This method is chosen over df["column"] = df["column"].map() since it will return "NaN" as the True and False values stem from an Excel function 
df["Gas"] = df["Gas"].astype("category") # Set the column as a category
df["Food"] = df["Food"].astype("category")
df["Clothing"] = df["Clothing"].astype("category")
df["Entertainment"] = df["Entertainment"].astype("category")
df["Investment"] = df["Investment"].astype("category")
df["Car_Maintenance"] = df["Car_Maintenance"].astype("category")
df["Deposit"] = df["Deposit"].astype("category")
new_categories = [0, 1] # Category map, False = 0; True = 1
df["Gas"] = df["Gas"].cat.rename_categories(new_categories)
df["Food"] = df["Food"].cat.rename_categories(new_categories)
df["Clothing"] = df["Clothing"].cat.rename_categories(new_categories)
df["Entertainment"] = df["Entertainment"].cat.rename_categories(new_categories)
df["Investment"] = df["Investment"].cat.rename_categories(new_categories)
df["Car_Maintenance"] = df["Car_Maintenance"].cat.rename_categories(new_categories)
df["Deposit"] = df["Deposit"].cat.rename_categories(new_categories)


# Converting columns from categories into integers for use in the logistic regression model
df["Gas"] = df.Gas.astype(int)
df["Food"] = df.Food.astype(int)
df["Clothing"] = df.Clothing.astype(int)
df["Entertainment"] = df.Entertainment.astype(int)
df["Car_Maintenance"] = df.Car_Maintenance.astype(int)
df["Investment"] = df.Investment.astype(int)
df["Deposit"] = df.Deposit.astype(int)
print(df)

# Printing summary statistics
df.describe()

# Finding correlations of all variables 
print(df.corr(numeric_only= True))

# Drawing Scatter plot for gas
plt.scatter(df.Amount_abs, df.Gas, color="Green")
plt.xlabel("Amount")
plt.ylabel("Gas (1=True, 0=False)")
plt.title("Expenditure Prediction for Gas")

# Fitting and initializing the logistical regression model for "gas"
x = df.Amount_abs
y = df.Gas
logreg = LogisticRegression(C=1.0, solver="lbfgs", multi_class="ovr")
X = x.values.reshape(-1, 1)
logreg.fit(X, y)

# Predictions for gas
print(logreg.predict([[110]])) 
print(logreg.predict([[100]]))
print(logreg.predict([[90]]))
print(logreg.predict([[80]]))
print(logreg.predict([[70]]))
print(logreg.predict([[60]]))
print(logreg.predict([[50]]))
print(logreg.predict([[40]]))
print(logreg.predict([[30]]))
print(logreg.predict([[20]]))
print(logreg.predict([[10]]))
# Issue within the dataset is that amounts for other variables contain amounts that are < $200 just like gas. A fix to this is to combine variables based on logical assumptions of necessary and discretionary spending, and investments and deposits. 

# Combine Gas, Food, and Car_Maintenance into Necessary Spending; Clothing, Miscellaneous, Entertainment into Discretionary Spending; Investment and Deposit
df = df.assign(NecessarySpending = df.Gas + df.Food + df.Car_Maintenance)
df = df.assign(DiscretionarySpending = df.Clothing + df.Entertainment)
df = df.assign(InvestmentDeposit = df.Investment + df.Deposit)
print(df)

# Drawing Scatter plot for Necessary Spending
plt.scatter(df.Amount_abs, df.NecessarySpending, color="Red")
plt.xlabel("Amount")
plt.ylabel("Probability of Necessary Spending")
plt.title("Transaction Prediction for Necessary Spending")

# Fitting and initializing the logistical regression model for Necessary Spending
x = df.Amount_abs
y2 = df.NecessarySpending
logreg = LogisticRegression(C=1.0, solver="lbfgs", multi_class="ovr")
X = x.values.reshape(-1, 1)
logreg.fit(X, y2)

# Predictions for Necessary Spending
print(logreg.predict([[200]])) 
print(logreg.predict([[110]])) 
print(logreg.predict([[100]]))
print(logreg.predict([[90]]))
print(logreg.predict([[80]]))
print(logreg.predict([[70]]))
print(logreg.predict([[60]]))
print(logreg.predict([[50]]))
print(logreg.predict([[40]]))
print(logreg.predict([[30]]))
print(logreg.predict([[20]]))
print(logreg.predict([[10]]))

# Drawing Scatter plot for Discretionary Spending
plt.scatter(df.Amount_abs, df.DiscretionarySpending)
plt.xlabel("Amount")
plt.ylabel("Probability of Discretionary Spending")
plt.title("Transaction Prediction for Discretionary Spending")

# Fitting and initializing the logistical regression model for Discretionary Spending
x = df.Amount_abs
y3 = df.DiscretionarySpending
logreg = LogisticRegression(C=1.0, solver="lbfgs", multi_class="ovr")
X = x.values.reshape(-1, 1)
logreg.fit(X, y3)

# Predictions for Discretionary Spending
print(logreg.predict([[200]])) 
print(logreg.predict([[110]])) 
print(logreg.predict([[100]]))
print(logreg.predict([[90]]))
print(logreg.predict([[80]]))
print(logreg.predict([[70]]))
print(logreg.predict([[60]]))
print(logreg.predict([[50]]))
print(logreg.predict([[40]]))
print(logreg.predict([[30]]))
print(logreg.predict([[20]]))
print(logreg.predict([[10]]))

# Drawing Scatter plot for Investment and Deposits
plt.scatter(df.Amount_abs, df.InvestmentDeposit, color="Black")
plt.xlabel("Amount")
plt.ylabel("Probability of Investment and Deposits")
plt.title("Transaction Prediction for Investment and Deposits")

# Fitting and initializing the logistical regression model for Investment and Deposits
x = df.Amount_abs
y4 = df.InvestmentDeposit 
logreg = LogisticRegression(C=1.0, solver="lbfgs", multi_class="ovr")
X = x.values.reshape(-1, 1)
logreg.fit(X, y4)

# Predictions for Investment and Deposits
print(logreg.predict([[1100]])) 
print(logreg.predict([[1000]]))
print(logreg.predict([[900]]))
print(logreg.predict([[800]]))
print(logreg.predict([[700]]))
print(logreg.predict([[600]]))
print(logreg.predict([[500]]))
print(logreg.predict([[400]]))
print(logreg.predict([[300]]))
print(logreg.predict([[200]]))
print(logreg.predict([[100]]))