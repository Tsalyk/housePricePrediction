import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns


df = pd.read_csv("Real estate.csv")

# correlation
plt.figure()
sns.heatmap(df.corr(), annot=True)

# density
sns.displot(x=df['Y house price of unit area'], kde=True, aspect=1.5, color="purple")
plt.xlabel("house price of unit area")

# plt.show()

# remove unneccessary column with n.o
df = df.iloc[:, 1:]

# split data
X, y = df.iloc[:, :-1], df.iloc[:, [-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# model fitting
model = LinearRegression()
model.fit(X_train, y_train)

# prediction
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# estimation
print(mean_squared_error(y_pred_test, y_test))
print(mean_absolute_error(y_pred_test, y_test))

plt.figure()
plt.scatter(y=y_test, x=y_pred_test)
plt.show()
