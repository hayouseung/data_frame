import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 로드
df = pd.read_csv('./data/7.lpg_leakage.csv')

# 데이터셋의 첫 몇 행 출력
print("First few rows of the dataset:")
print(df.head())

# 데이터셋의 기본 통계량 출력
print("\nBasic statistics of the dataset:")
print(df.describe())

# 결측값 확인
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# 각 변수의 분포 시각화 및 저장
plt.figure(figsize=(15, 10))
df.hist(bins=30, figsize=(15, 10), layout=(3, 3))
plt.tight_layout()
plt.savefig('feature_distribution.png')

# 상관 행렬 시각화 및 저장
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('./7.lpg_leakage.png')

# 목표 변수(LPG_Leakage)의 균형 확인
print("\nBalance of the target variable (LPG_Leakage):")
print(df['LPG_Leakage'].value_counts())

# 특징 변수 정규화 (목표 변수 제외)
features = df.drop(columns=['LPG_Leakage'])
normalized_features = (features - features.mean()) / features.std()
df_normalized = pd.concat([normalized_features, df['LPG_Leakage']], axis=1)

# 정규화된 데이터셋의 첫 몇 행 출력
print("\nFirst few rows of the normalized dataset:")
print(df_normalized.head())

# 정규화된 데이터셋을 새로운 CSV 파일로 저장
df_normalized.to_csv('./7.lpg_leakage_normalized.csv', index=False)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 로드 및 전처리
df = pd.read_csv('./data/7.lpg_leakage.csv')
features = df.drop(columns=['LPG_Leakage'])
normalized_features = (features - features.mean()) / features.std()
df_normalized = pd.concat([normalized_features, df['LPG_Leakage']], axis=1)

# 데이터셋을 학습 데이터와 테스트 데이터로 분리
X = df_normalized.drop(columns=['LPG_Leakage'])
y = df_normalized['LPG_Leakage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 학습 및 평가
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg)}")
print(f"Precision: {precision_score(y_test, y_pred_log_reg)}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg)}")
print(f"F1 Score: {f1_score(y_test, y_pred_log_reg)}")

# 랜덤 포레스트 모델 학습 및 평가
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf_clf = rf_clf.predict(X_test)
print("\nRandom Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_clf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf_clf)}")
print(f"Recall: {recall_score(y_test, y_pred_rf_clf)}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf_clf)}")

# SVM 모델 학습 및 평가
svm_clf = SVC(probability=True)
svm_clf.fit(X_train, y_train)
y_pred_svm_clf = svm_clf.predict(X_test)
print("\nSupport Vector Machine Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm_clf)}")
print(f"Precision: {precision_score(y_test, y_pred_svm_clf)}")
print(f"Recall: {recall_score(y_test, y_pred_svm_clf)}")
print(f"F1 Score: {f1_score(y_test, y_pred_svm_clf)}")

# 혼동 행렬 시각화 및 저장
conf_matrix = confusion_matrix(y_test, y_pred_rf_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('./confusion_matrix_rf.png')

# ROC 곡선 시각화 및 저장 (랜덤 포레스트 모델 기준)
y_prob_rf_clf = rf_clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_rf_clf)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Random Forest Classifier')
plt.legend(loc="lower right")
plt.savefig('./roc_curve_rf.png')

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 데이터셋 로드
df = pd.read_csv('./data/7.lpg_leakage.csv')

# 특징 변수와 목표 변수 분리
X = df.drop(columns=['LPG_Leakage'])
y = df['LPG_Leakage']

# KFold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 로지스틱 회귀 모델 교차 검증
log_reg = LogisticRegression()
log_reg_scores = cross_val_score(log_reg, X, y, cv=kf, scoring='accuracy')
print("Logistic Regression Cross-Validation Scores:", log_reg_scores)
print("Logistic Regression Mean Accuracy:", log_reg_scores.mean())

# 랜덤 포레스트 모델 교차 검증
rf_clf = RandomForestClassifier()
rf_clf_scores = cross_val_score(rf_clf, X, y, cv=kf, scoring='accuracy')
print("Random Forest Classifier Cross-Validation Scores:", rf_clf_scores)
print("Random Forest Classifier Mean Accuracy:", rf_clf_scores.mean())

# SVM 모델 교차 검증
svm_clf = SVC()
svm_clf_scores = cross_val_score(svm_clf, X, y, cv=kf, scoring='accuracy')
print("Support Vector Machine Classifier Cross-Validation Scores:", svm_clf_scores)
print("Support Vector Machine Classifier Mean Accuracy:", svm_clf_scores.mean())