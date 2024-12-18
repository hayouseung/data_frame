# 데이터 처리를 위한 pandas 라이브러리 불러오기
import pandas as pd

import matplotlib

import matplotlib.pyplot as pyplot
# 수학 계산 및 행렬 연산을 위해 numpy 라이브러리 불러오기
import numpy as np
import matplotlib.pyplot as plt

# 데이터 파일 이름 지정: 상대경로로 데이터 파일의 위치와 데이터를 읽고 이를 데이터프레임으로 저장 및 출력하기 위한 가정
filename = "../data/1_pima.csv"

# 데이터에 사용될 컬럼 이름 지정
column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# 데이터 파일을 읽어와 데이터 프레임에 저장하고 데이터프레임을 출력 -> 데이터 분석에 이용할 컬럼 이름 지정
data = pd.read_csv(filename, names=column_names)

# 데이터의 상관계수 계산 및 출력
correlations = data.corr(method='pearson')
print(correlations)
# 데이터 상관관계 csv 파일로 저장
correlations.to_csv("./results/correlation_coefficient.csv")


# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(column_names)
ax.set_yticklabels(column_names)

# save plot
plt.savefig("./results/correlation_plot.png")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# 파일 불러오기
filename = '../data/1_pima.csv'

# 열 이름 설정
column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# 데이터 읽어오기
data = pd.read_csv(filename, names=column_names)

# 예측을 위한 데이터 나누기
X = data.iloc[:, :-1].values    # 입력 변수 선택 (class 제외)
y = data.iloc[:, -1].values     # 출력 변수인 class 선택

# Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 학습된 모델을 사용하여 테스트 데이터에 대한 예측 수행
y_pred = model.predict(X_test)
print(y_pred)

# 예측값을 0 또는 1로 변환 (임계값 설정 적용)
y_pred_binary = (y_pred > 0.5).astype(int)
print(y_pred_binary)

# 예측 정확도 계산
accuracy = accuracy_score(y_test, y_pred_binary)

# 결과 출력
print("----------------------------------")
print("Actual Values:", y_test)
print("Predicted Values:", y_pred_binary)
print("----------------------------------")
print("Accuracy:", accuracy)

import matplotlib.pyplot as plt

# 그래프 설정
plt.figure(figsize=(10, 5))

# 실제값과 예측값 시각화
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(y_pred_binary)), y_pred_binary, color='red', label='Predicted Values', marker='x')

plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Data Index')
plt.ylabel('Class (0 or 1)')
plt.legend()

# 그래프 저장
plt.savefig('./results/linear_regression.png')
