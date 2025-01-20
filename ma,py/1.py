#LogisticRegression 모델 정의하고 학습
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)

#KNN으로 모델을 정의하고 학습 (n_neighbors=5)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#Decision Tree로 모델을 정의하고 학습(max_depth=10, random_state=42)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

#RandomForest로 모델을 정의하고 학습(n_estimators=3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=3, random_state=42)
rfc.fit(X_train, y_train)

#XGBoost로 모델을 정의하고 학습(n_estimators=3, random_state=42)
#!pip install xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=3, random_state=42)  
xgb.fit(X_train, y_train)

#Light GBM으로 모델을 정의하고 학습(n_estimators=3, random_state=42)
#!pip install lightgbm
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(n_estimators=3, random_state=42)  
lgbm.fit(X_train, y_train)