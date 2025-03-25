# 🧳 여행 만족도 예측 시스템

여행자의 특성과 여행 스타일을 바탕으로 다양한 여행지에 대한 예상 만족도를 제공하는 시스템입니다. 사용자와 비슷한 성향을 가진 다른 여행자들의 실제 경험 데이터를 분석하여 최적의 여행지를 추천합니다.

## 📝 개요

이 프로젝트는 한국관광공사의 여행 데이터를 활용하여 개발된 AI 기반 여행지 추천 시스템입니다. CatBoost 알고리즘을 사용하여 여행자 특성, 선호도, 여행 스타일에 따른 만족도를 예측합니다.

## 🌟 주요 기능

- **개인화된 여행지 추천**: 성별, 연령대, 여행 동기, 여행 목적, 동반자 수 등 개인 특성 기반 추천
- **여행 스타일 분석**: 8가지 여행 스타일 측정 및 시각화
- **사용자 프로필 시각화**: 레이더 차트를 통한 사용자 여행 스타일 시각화
- **추천 결과 등급화**: 예상 만족도에 따른 등급 및 설명 제공
- **직관적인 UI**: 시각적으로 풍부한 결과 제공

## 🔧 설치 및 실행 방법

### 요구 사항

- Python 3.7 이상
- pip 패키지 관리자

### 설치 방법

1. 저장소를 클론합니다.

   ```bash
   git clone <repository-url>
   cd travel-satisfaction-prediction
   ```

2. 필요한 패키지를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

### 실행 방법

1. Streamlit 앱을 실행합니다.

   ```bash
   streamlit run app.py
   ```

2. 웹 브라우저에서 http://localhost:8501 로 접속합니다.

## 📂 파일 구조

```
├── app.py                   # 메인 애플리케이션 파일
├── constants.py             # 상수 및 매핑 정의
├── model.py                 # 모델 관리 및 예측 기능
├── ui.py                    # UI 컴포넌트 클래스
├── utils.py                 # 유틸리티 함수
├── data/
│   └── df_filter.csv        # 학습 및 예측에 사용되는 데이터
├── catboost_model.cbm       # 학습된 CatBoost 모델
├── travel_style_analysis.csv    # 여행 스타일 분석 결과
├── travel_style_distribution.png # 여행 스타일 분포 이미지
└── requirements.txt         # 필요한 패키지 목록
```

## 🔄 모듈 구성

- **constants.py**: 시스템에서 사용하는 상수 및 매핑 정보 정의
- **model.py**: 모델 관리, 데이터 전처리, 예측 기능 구현
- **ui.py**: UI 컴포넌트, 시각화, 페이지 레이아웃 관리
- **utils.py**: 한글 폰트 설정, 만족도 등급 계산 등 유틸리티 함수
- **app.py**: 메인 애플리케이션 진입점

## 📊 데이터 구조

- **GENDER**: 성별 (남/여)
- **AGE_GRP**: 연령대 (20, 30, 40, 50, 60, 70)
- **TRAVEL_STYL_1~8**: 여행 스타일 (1-7 척도)
  - TRAVEL_STYL_1: 자연 vs 도시
  - TRAVEL_STYL_2: 숙박 vs 당일
  - TRAVEL_STYL_3: 새로운지역 vs 익숙한지역
  - TRAVEL_STYL_4: 편하지만 비싼 숙소 vs 불편하지만 저렴한 숙소
  - TRAVEL_STYL_5: 휴양/휴식 vs 체험활동
  - TRAVEL_STYL_6: 잘 알려지지 않은 방문지 vs 알려진 방문지
  - TRAVEL_STYL_7: 계획에 따른 여행 vs 상황에 따른 여행
  - TRAVEL_STYL_8: 사진촬영 중요하지 않음 vs 사진촬영 중요
- **TRAVEL_MOTIVE_1**: 여행 동기 (10개 카테고리)
- **TRAVEL_COMPANIONS_NUM**: 동반자 수
- **TRAVEL_MISSION_INT**: 여행 목적 (5개 카테고리)
- **VISIT_AREA_NM**: 방문 지역명
- **DGSTFN**: 여행 만족도 (1-5 척도)

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
