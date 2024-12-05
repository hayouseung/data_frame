#1
import pandas as pd
filename = "../data/1_pima.csv"
column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=column_names)
correlations = data.corr(method='pearson')
print(correlations)

#2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
filename = "../data/1_pima.csv"
column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=column_names)
correlations = data.corr(method='pearson')

#3
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(column_names)
ax.set_yticklabels(column_names)
# 매트릭스 안에 텍스트 넣는 소스코드 추가했습니다.
for (i, j), value in np.ndenumerate(correlations):
    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")
plt.savefig("./correlation_plot.png")

#4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
filename = "../data/1_pima.csv"
column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=column_names)
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

#5
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

#6
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(Y_pred)
Y_pred_binary = (Y_pred > 0.5).astype(int)
print(Y_pred_binary)
accuracy = accuracy_score(Y_test, Y_pred_binary)

#7
print("---------------------")
print("Actual Values:", Y_test)
print("Predicted Values:", Y_pred_binary)
print("---------------------")
print("Accuracy:", accuracy)

#8
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(Y_pred_binary)), Y_pred_binary, color='red', label='predicted Values', marker='x')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Data Index')
plt.ylabel('Class (0 or 1)')
plt.legend()
plt.savefig("./linear_regression.png")