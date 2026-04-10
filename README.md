# IMBK_Bank_Customer_Churn_ML

## 1. 프로젝트 명 : 고객이탈분류 ML 및 인사이트 분석
| 핵심 | 내용 |
| ------ | ------ | 
| 프로젝트 명 | 고객이탈분류 ML 및 인사이트 분석 |
| 기간 | 2026.04.10 |
| 데이터 출처 | kaggle Bank Customer Churn Dataset (row: 10000, col:12) |

# 2. 기술 스택

Programming Language: Python

Data Processing: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, PyCaret

Hyperparameter Optimization: Optuna



# 3. 데이터 전처리

데이터 전처리 단계에서는 모델 학습에 적합한 형태로 데이터를 가공하는 과정을 거쳤습니다. 먼저, 분석에 불필요한 변수인 customer_id 컬럼을 제거하여 학습 시 불필요한 노이즈가 반영되지 않도록 하였습니다 (df.drop(columns=['customer_id'])). 이후, 범주형 변수인 country와 gender는 머신러닝 모델이 처리할 수 있도록 Label Encoding을 적용하여 수치형 데이터로 변환하였습니다 (LabelEncoder()). 또한, 데이터의 결측치 여부와 자료형을 확인하여 전반적인 데이터 상태를 점검하였으며 (df.isnull().sum(), df.info()), 모델의 일반화 성능을 확보하기 위해 타겟 변수인 churn의 클래스 비율을 유지한 상태로 학습용과 검증용 데이터로 분리하는 과정을 거쳤습니다 (train_test_split(..., stratify=df['churn'])).

[ 사용된 코드 ]

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

1. 데이터 로드
   
df = pd.read_csv('Bank Customer Churn Prediction.csv')

3. 불필요한 컬럼 제거
   
df = df.drop(columns=['customer_id'], errors='ignore')

4. 범주형 변수 인코딩
   
le_country = LabelEncoder()

le_gender = LabelEncoder()

df['country'] = le_country.fit_transform(df['country'])

df['gender'] = le_gender.fit_transform(df['gender'])

4. 결측치 및 데이터 구조 확인
   
print(df.isnull().sum())

print(df.info())

5. 학습/검증 데이터 분리 (Stratified Split)
   
train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['churn']
)


## 4. EDA 및 해석

# 결측치 확인

<img width="183" height="259" alt="image" src="https://github.com/user-attachments/assets/568fd250-490c-4403-a555-11472f169943" />

- 이상 없었습니다

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/8eba45b3-e5a3-4a2e-938b-ff71afbac19e" />

- 먼저, Churnd 에서

본 프로젝트는 은행 고객 이탈 예측을 주제로 Python 기반 데이터 분석 및 머신러닝 모델링 전 과정을 수행한 프로젝트입니다. Pandas와 NumPy를 활용해 데이터를 전처리하고, Seaborn과 Matplotlib을 통해 고객 특성별 이탈 패턴을 분석하여 주요 인사이트를 도출하였습니다. 이후 PyCaret을 활용해 다양한 분류 모델을 비교하여 상위 모델을 선정하고, Optuna 기반 하이퍼파라미터 튜닝을 통해 성능을 개선하였으며, Stacking Ensemble 기법으로 최종 모델의 예측력을 향상시켰습니다. 또한 SHAP을 활용하여 모델의 의사결정 과정을 해석하고 주요 변수의 영향력을 분석함으로써, 단순 예측을 넘어 실질적인 고객 유지 전략 수립에 기여할 수 있도록 설계하였습니다.
