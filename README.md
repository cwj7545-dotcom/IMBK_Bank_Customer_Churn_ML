# IMBK_Bank_Customer_Churn_ML

# 1. 프로젝트 명 : 고객이탈분류 ML 및 인사이트 분석
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


# 4. EDA 및 해석

## 결측치 확인

<img width="183" height="259" alt="image" src="https://github.com/user-attachments/assets/568fd250-490c-4403-a555-11472f169943" />

- 이상 없었습니다

## EDA 1

### 핵심 차트

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/8eba45b3-e5a3-4a2e-938b-ff71afbac19e" />

분석 결과: 활동 회원 여부에 따른 이탈 현황 (Churn by Active Member Status)
ActiveMember 변수를 기준으로 고객의 이탈(Churn) 여부를 시각화하여 비교 분석을 진행했습니다.

|회원 상태| 유지 (Stayed) | 이탈 (Left) | 이탈률 (Churn Rate) |
| ------ | ------ | ------ | ------ |
| 비활동 회원(Inactive) | "3,547명" | "1,302명" | 약 26.8% |
| 활동 회원 (Active) | "4,416명" | 735명  |약 14.3% |

### 인사이트

- 활동 상태와 이탈의 상관관계: 활동 회원(Active) 그룹의 이탈률은 약 14%인 반면, 비활동 회원(Inactive) 그룹의 이탈률은 약 27%로 2배 가까이 높게 나타났습니다.

- 잠재적 이탈 위험군 식별: 현재 서비스 내에서 활동이 적은(Inactive) 고객일수록 실제 서비스 해지로 이어질 가능성이 매우 높음을 시사합니다.

- 비즈니스 전략 제언: 전체 이탈자의 상당수(약 64%)가 비활동 회원군에서 발생하고 있습니다. 따라서 이탈 방지를 위해 비활동 고객을 다시 활동 상태로 전환시키기 위한 재참여(Re-engagement) 마케팅 캠페인이 우선적으로 필요할 것으로 보입니다.

## EDA 2

### 핵심차트

<img width="700" height="507" alt="image" src="https://github.com/user-attachments/assets/b9ddc628-ac34-43ed-bd1c-56d800e3ba77" />

분석 결과: 국가별 고객 분포 (Distribution of Customers by Country)
데이터셋에 포함된 세 국가(프랑스, 독일, 스페인)의 고객 비중을 확인하기 위해 countplot을 활용한 시각화를 진행했습니다.

| 국가 | 내용 |
| ---- | ---- |
| France (프랑스) | 약 5,000명으로 전체 데이터에서 가장 큰 비중(약 50%)을 차지하고 있습니다. |
| Germany (독일) | 약 2,500명 수준입니다. |
| Spain (스페인) | 독일과 유사한 약 2,500명 수준으로 확인됩니다. |

### 인사이트

- 데이터 불균형 확인: 프랑스 고객의 수가 독일과 스페인을 합친 수와 거의 대등할 정도로 많습니다. 이는 모델링 과정에서 특정 국가에 편향된 학습이 일어날 수 있음을 시사하며, 향후 국가별 이탈률(Churn Rate)을 추가 분석할 필요가 있습니다.

- 시장 중요도: 단순 고객 수 측면에서는 프랑스가 가장 핵심적인 시장이지만, 상대적으로 표본이 적은 독일과 스페인 시장의 데이터 특성을 면밀히 파악하는 것이 중요합니다.

# EDA 3

### 핵심 차트

<img width="1584" height="1141" alt="image" src="https://github.com/user-attachments/assets/8cb176f1-de21-4d33-94d6-08ea9d1acd46" />

분석 결과 : 독일 시장의 변수별 이탈률 트렌드 (Trend Analysis in Germany)
단순한 분포를 넘어, 특정 변수의 구간(Binning)에 따른 이탈률(Average Churn Rate) 변화를 추적하여 독일 시장 고객의 이탈 특성을 도출했습니다.

### 인사이트

독일 시장에서 중장년층 고객의 이탈 관리가 비즈니스의 가장 시급한 과제임을 보여줍니다. 60대 이후 급격히 감소하는 것은 은퇴 및 계좌 유지 성향과 관련이 있을 수 있습니다.

1. 연령대별 이탈률 (Age)
   
| 특이점 | 인사이트 |
| ----- | ----- |
| 40대 중반부터 이탈률이 급격히 상승하여 50~55세 구간에서 약 70%로 정점 기록 | 독일 시장 내 중장년층 고객 관리가 최우선 과제임. 60대 이후의 감소세는 은퇴 및 자산 유지 성향과 관련이 있을 것으로 분석됨 |

2. 신용 점수별 이탈률 (Credit)
   
| 특이점 | 인사이트 |
| ----- | ----- |
| 신용 점수 400점 미만 저신용 구간에서 이탈률이 100%에 육박함 | 극단적 저신용 상태는 매우 강력한 이탈 예측 지표(Predictor)임. 450점 이상부터는 이탈률이 약 0.3 내외로 안정화되는 경향을 보임 |

3. 거래 기간별 이탈률 (Tenure)
   
| 특이점 | 인사이트 |
| ----- | ----- |
| 가입 후 1년 차 고객의 이탈률이 약 40%로 전 구간 중 가장 높게 나타남 | 서비스 안착에 실패하는 초기 이탈 현상(Early Churn) 관찰됨. 신규 가입자 대상 온보딩 프로그램 및 초기 리워드 강화가 필요함 |

4. 급여 수준 별 이탈률
   
| 특이점 | 인사이트 |
| ----- | ----- |
|소득 하위 10%(Decile 0) 그룹에서 이탈률이 가장 높으며, 소득 수준 상승에 따라 완만한 하향 곡선 형성 | 저소득 그룹일수록 가격 민감도가 높고 서비스 교체 성향이 강할 가능성이 큼. 경제적 여건에 따른 맞춤형 금융 상품 제안이 요구됨 |

# AUTO ML

## PYCARET 을 통한, 모델 성능 비교 (F1 Score)

### 1. PyCaret setup
exp = setup(
    data=df,
    target='churn',
    session_id=42,
    fix_imbalance=False,   # 속도 위해 False
    fold=3,                # 속도 위해 3-fold
    verbose=False
)

### 2. Optuna를 통한 튜닝 및 머신 성능측정

[ 코드]

top4 모델 Optuna로 각각 5번 튜닝

tuned_top4_models = []

print("\n상위 4개 모델 튜닝 시작...\n")

for i, model in enumerate(top4_models):
    print(f"{i+1}번째 모델 튜닝 중: {model}")
    
    # optuna에서 찾고, 최적화 방식은 f1 score로 합니다, 횟수는 최대한 빠르게 하기 위해서 5회로 했습니다, fold수도 역시 3회로했습니다
    
    tuned_model = tune_model(
        model,
        search_library='optuna',
        optimize='F1',
        n_iter=5,              # 5번만 탐색
        choose_better=True,
        fold=3,
        verbose=False
    )
    
    tuned_top4_models.append(tuned_model)

<img width="816" height="483" alt="image" src="https://github.com/user-attachments/assets/2e3d8f49-5a7d-4ef2-bad0-535d3a80f0f4" />

### 인사이트

| Model | Accuracy | AUC |	Recall  | Precision | F1 | 특징 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| Gradient Boosting (GBC) | 0.8607 | 0.8578 | 0.4551 | 0.7671 | 0.5705 | 최고 정확도 및 정밀도 보유 |
| LightGBM | 0.8550 | 0.8483 | 0.4811 | 0.7145 | 0.5746 | F1-Score 기준 최적의 균형 |
| CatBoost | 0.8577 | 0.8568 | 0.4685 | 0.7373 | 0.5726 | 전반적으로 매우 안정적인 수치 |
| Decision Tree (DT) | 0.7791 | 0.6745 | 0.4979 | 0.4609 | 0.4784 | Recall(재현율)이 가장 높음 |



### 모델 선택 이유

성능 비교 결과 수치상으로는 4위를 기록했으나, 본 프로젝트의 최종 예측 모델로 Random Forest를 채택하였습니다. 그 이유는 다음과 같습니다.

분석 프로세스의 일관성 및 숙련도: 이번 분석 과정 전반에서 Random Forest 모델을 지속적으로 활용하며 하이퍼파라미터 튜닝 및 성능 최적화 경험을 쌓았습니다. 모델의 특성을 깊이 이해하고 있는 만큼, 결과값에 대한 신뢰도를 높이고 예외 상황에 유연하게 대응하기 위해 해당 모델을 최종 선정했습니다.

성능 차이의 유의성 검토: 1위 모델(GBC, 0.8607)과 Random Forest(0.8580)의 성능 차이는 약 0.3% 미만으로 매우 적습니다. 이러한 미세한 수치 차이보다는 모델의 안정성과 해석 가능성(Explainability)을 우선순위에 두었습니다.

SHAP Value 분석과의 연계: Random Forest는 다수의 의사결정 나무를 결합한 모델로, 이후 진행할 SHAP Value 분석에서 변수 간의 상호작용을 직관적으로 파악하기에 매우 적합한 구조를 가지고 있습니다.

| 항목 | 내용 | 비고|
| ----- | ----- | ----- |
| 최종 모델 | Random Forest Classifier | 앙상블 기반의 높은 안정성 보유 |
| 선정 근거 1 | 분석 숙련도 및 일관성 | 프로젝트 전반에서 활용하며 모델 제어 및 튜닝 역량 확보
| 선정 근거 2 | 성능 차이 미미 | 1위(GBC) 대비 정확도 차이가 0.3% 미만으로 유의미한 성능 유지 |
| 선정 근거 3 | XAI(해석 가능성) 연계 | SHAP 분석을 통해 모델의 예측 근거를 시각화하기에 최적화된 구조 |

# SHAP value

## SHAP value 분석

    explainer = shap.TreeExplainer(rf_model)
    
    shap_values = explainer.shap_values(X_test_sample)

    print("X_test_sample shape:", X_test_sample.shape)

    # shap_values 형태 확인

    # shap values가 list인지 확인
    
    if isinstance(shap_values, list):
        print("shap_values is list")
        print("각 클래스 shap shape:", [sv.shape for sv in shap_values])
        shap_to_plot = shap_values[1]   # churn=1 클래스
        
    else:
        print("shap_values shape:", shap_values.shape)

        # SHAP 결과가 3차원이면 chrun = 1 클래스만 꺼냅니다
        
        if len(shap_values.shape) == 3:
            shap_to_plot = shap_values[:, :, 1]   
        else:
            shap_to_plot = shap_values

    shap.summary_plot(shap_to_plot, X_test_sample)

<img width="759" height="614" alt="image" src="https://github.com/user-attachments/assets/897a9ea5-64de-47f6-a2c5-417cd3cec0ef" />

## 분석 결과

사용변수

| 변수명 | 분석 결과 및 비즈니스 인사이트 |
| ----- | ----- |
| Age (나이) | "가장 지배적인 변수입니다. 시각화 결과 나이가 많을수록(Red) SHAP Value가 양(+)의 방향으로 나타나 이탈 확률이 급격히 높아집니다. 반면, 젊은 층은 이탈 가능성이 상대적으로 낮아 안정적인 고객군임을 시사합니다." |
| Products Number | "비선형적(Non-linear) 패턴이 관찰됩니다. 일반적으로 상품 수가 적을수록 이탈 위험이 크지만, 특정 임계값을 넘어 상품 수가 과도하게 많은 고객군에서도 이탈 위험이 급증하는 양상을 보입니다. 이는 상품 가입 유도뿐만 아니라 가입 후 관리가 중요함을 의미합니다." |
| Active Member | 활동성 여부는 이탈과 강한 음(-)의 상관관계를 가집니다. 활동 회원(Red)일수록 SHAP Value가 좌측으로 형성되어 이탈 가능성을 크게 낮추는 핵심 방어 요인으로 작용합니다. |
| Country (Germany) | "독일 국적 고객(Red)의 경우 타 국가 대비 SHAP Value가 우측에 집중되어 있습니다. 이는 독일 시장 고객의 이탈 성향이 유독 강함을 나타내며, 해당 지역의 경쟁 환경이나 서비스 만족도에 대한 로컬 분석이 필요함을 보여줍니다." |

기타 변수 및 한계점 (Minor Features)

| 변수명 | 분석 결과 및 비즈니스 인사이트 |
| ----- | ----- |
| Balance (잔고) | 잔고가 높을수록 이탈 위험이 소폭 상승하는 경향이 있으나, SHAP Value가 0 근처에 밀집되어 있어 영향력은 상위 변수들에 비해 제한적입니다. |
| Gender & Others | 성별(Gender), 신용 점수(Credit Score), 추정 급여(Estimated Salary) 등은 SHAP Value가 0에 수렴하고 있습니다. 이는 해당 모델이 이탈 여부를 판단할 때 위 변수들을 주요 결정 근거로 삼지 않았음을 의미합니다. |



# 최종 인사이트

## 데이터 기반 최종 비즈니스 전략 제안 (Actionable Insights)

본 프로젝트를 통해 도출된 핵심 지표를 바탕으로, 고객 이탈 방지를 위한 3대 집중 관리 전략을 제안합니다.

1. [High-Risk] 중장년층 고객 타겟 맞춤형 유지 전략
   
현황: SHAP 분석 결과 Age가 이탈의 가장 큰 원인이며, 특히 독일 지역의 50~55세 구간 고객은 약 70%의 매우 높은 이탈률을 보입니다.

전략: 자산 관리 프로그램 강화: 은퇴 준비 시점인 해당 연령대의 특성에 맞춰 연금 상품이나 안정적인 자산 운용 컨설팅 서비스를 제공하여 락인(Lock-in) 효과를 유도합니다.

로열티 프로그램 도입: 장기 거래 고객이 많은 연령대인 만큼, 거래 기간(Tenure)에 따른 우대 금리나 수수료 감면 혜택을 강화합니다.

2. [Product] 상품 가입 포트폴리오 최적화
   
현황: Products Number 분석 결과, 상품이 너무 적거나 반대로 너무 과도한 고객층에서 비선형적인 이탈 패턴이 발견되었습니다.

전략: Cross-selling 가이드라인 수립: 단순히 상품 수를 늘리는 것이 아니라, 고객의 라이프사이클에 맞는 '핵심 상품 조합(예: 급여계좌 + 적금 + 신용카드)'을 설계하여 제안합니다.

다상품 가입자 특별 케어: 상품을 많이 보유한 고객은 서비스 복잡도에 지칠 수 있으므로, 통합 관리 UI 제공이나 전담 상담 서비스를 통해 관리 편의성을 증대시킵니다.

# REFERENCE

Data Source

Kaggle: Bank Customer Churn Prediction Dataset

Tools & Libraries
Language: Python (3.10.20)

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: PyCaret (AutoML), Scikit-learn

XAI: SHAP (SHapley Additive exPlanations)

Methodology & Documentation

Model Interpretation: SHAP Official Documentation

AutoML Framework: PyCaret Documentation

Theory: Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in neural information processing systems 30 (2017).

## 요약

본 프로젝트는 은행 고객 이탈 예측을 주제로 Python 기반 데이터 분석 및 머신러닝 모델링 전 과정을 수행한 프로젝트입니다. Pandas와 NumPy를 활용해 데이터를 전처리하고, Seaborn과 Matplotlib을 통해 고객 특성별 이탈 패턴을 분석하여 주요 인사이트를 도출하였습니다. 이후 PyCaret을 활용해 다양한 분류 모델을 비교하여 상위 모델을 선정하고, Optuna 기반 하이퍼파라미터 튜닝을 통해 성능을 개선하였으며, Stacking Ensemble 기법으로 최종 모델의 예측력을 향상시켰습니다. 또한 SHAP을 활용하여 모델의 의사결정 과정을 해석하고 주요 변수의 영향력을 분석함으로써, 단순 예측을 넘어 실질적인 고객 유지 전략 수립에 기여할 수 있도록 설계하였습니다.
