import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple
import platform
import matplotlib.font_manager as fm
import time
import base64
from PIL import Image
import io

# 페이지 설정
st.set_page_config(
    page_title="여행 만족도 예측 시스템",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 앱 스타일 설정
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #26A69A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .highlight {
        font-weight: bold;
        color: #FF5722;
    }
    .recommendation-item {
        padding: 0.8rem;
        border-radius: 5px;
        background-color: #e3f2fd;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f0f0;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    /* 사이드바 스타일 */
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    /* 선택 슬라이더 스타일 */
    div[data-baseweb="select-slider"] {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 상수 정의
CATEGORICAL_FEATURES = [
    'GENDER', 
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 
    'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 
    'TRAVEL_STYL_7', 'TRAVEL_STYL_8', 
    'TRAVEL_MOTIVE_1', 
    'TRAVEL_MISSION_INT', 
    'VISIT_AREA_NM'
]

NUMERICAL_FEATURES = ['AGE_GRP', 'TRAVEL_COMPANIONS_NUM']

# 여행 스타일 정의
TRAVEL_STYLE_MAPPING = {
    'TRAVEL_STYL_1': ('자연', '도시'),
    'TRAVEL_STYL_2': ('숙박', '당일'),
    'TRAVEL_STYL_3': ('새로운지역', '익숙한지역'),
    'TRAVEL_STYL_4': ('편하지만 비싼 숙소', '불편하지만 저렴한 숙소'),
    'TRAVEL_STYL_5': ('휴양/휴식', '체험활동'),
    'TRAVEL_STYL_6': ('잘 알려지지 않은 방문지', '알려진 방문지'),
    'TRAVEL_STYL_7': ('계획에 따른 여행', '상황에 따른 여행'),
    'TRAVEL_STYL_8': ('사진촬영 중요하지 않음', '사진촬영 중요')
}

# 여행 동기 매핑
TRAVEL_MOTIVE_MAPPING = {
    '1': '일상 탈출, 지루함 탈피',
    '2': '휴식, 피로 해소',
    '3': '동반자와의 친밀감 증진',
    '4': '자아 찾기, 자기성찰',
    '5': 'SNS 등 과시',
    '6': '운동, 건강 증진',
    '7': '새로운 경험 추구',
    '8': '역사 탐방, 문화 체험',
    '9': '특별한 목적(칠순, 신혼여행 등)',
    '10': '기타'
}

# 여행 목적 매핑
TRAVEL_MISSION_MAPPING = {
    '1': '휴식/힐링',
    '2': '문화체험',
    '3': '관광/명소방문',
    '4': '음식/맛집',
    '5': '쇼핑/레저'
}

# 연령대 매핑
AGE_GROUP_MAPPING = {
    20: "20대",
    30: "30대",
    40: "40대",
    50: "50대",
    60: "60대",
    70: "70대 이상"
}

# 한글 폰트 설정
def set_korean_font():
    system_name = platform.system()
    if system_name == 'Windows':
        # 윈도우에서 많이 사용되는 한글 폰트들
        for font_name in ['Malgun Gothic', '맑은 고딕', 'NanumGothic', 'Gulim', '굴림', 'Dotum', '돋움', 'Batang', '바탕']:
            try:
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                if not 'DejaVuSans.ttf' in font_path:  # 기본 폰트가 아닐 경우
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                    return
            except:
                continue
        st.warning("사용 가능한 한글 폰트를 찾을 수 없습니다.")
    else:
        # macOS 또는 Linux의 경우
        try:
            plt.rc('font', family='AppleGothic' if system_name == 'Darwin' else 'NanumGothic')
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        except:
            st.warning(f"{system_name} 시스템에서 한글 폰트 설정에 실패했습니다.")

# 페이지 구분 함수
def show_intro_page():
    """소개 페이지를 표시합니다."""
    st.markdown('<h1 class="main-header">🧳 여행 만족도 예측 시스템</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🌏 시스템 소개</h3>
            <p class="info-text">
                이 시스템은 여행자의 특성과 선호도를 바탕으로 다양한 여행지에 대한 예상 만족도를 제공합니다.
                인공지능 모델을 통해 사용자와 비슷한 성향을 가진 다른 여행자들의 실제 경험 데이터를 분석하여
                가장 만족스러울 것으로 예상되는 여행지를 추천해 드립니다.
            </p>
            <h4>사용 방법</h4>
            <ol>
                <li>좌측 사이드바에서 여행자 정보와 선호도를 입력합니다.</li>
                <li>여행 스타일, 여행 동기, 동반자 수, 여행 목적 등을 선택합니다.</li>
                <li>시스템이 자동으로 여행지별 예상 만족도를 계산합니다.</li>
                <li>상위 추천 여행지와 상세 분석 결과를 확인합니다.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>📊 데이터 기반 추천</h3>
            <p class="info-text">
                본 시스템은 한국관광공사의 관광 데이터를 기반으로 학습된 CatBoost 알고리즘을 사용합니다.
                실제 여행자들의 만족도 평가를 학습하여 사용자에게 적합한 여행지를 추천합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🔍 여행 스타일 이해하기</h3>
            <p class="info-text">
                여행 스타일은 여행 만족도에 큰 영향을 미치는 요소입니다.
                자신의 여행 스타일을 파악하고 입력하면 더 정확한 추천을 받을 수 있습니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 여행 스타일 분석 결과 불러오기
        try:
            style_analysis = pd.read_csv('travel_style_analysis.csv')
            with st.expander("📊 여행 스타일 분석 결과"):
                st.dataframe(style_analysis)
                
            if st.checkbox("여행 스타일 분포 보기"):
                set_korean_font()  # 한글 폰트 설정
                try:
                    st.image('travel_style_distribution.png')
                except:
                    st.warning("여행 스타일 분포 이미지를 찾을 수 없습니다.")
        except Exception as e:
            st.info("여행 스타일 분석 결과를 불러올 수 없습니다. 시스템을 계속 사용할 수 있습니다.")
        
        st.markdown("""
        <div class="card">
            <h3>💡 시작하기</h3>
            <p class="info-text">
                좌측 사이드바에서 여행자 정보를 입력하고, 상단 탭에서 <span class="highlight">'여행지 추천'</span> 탭을 선택하여 
                추천 결과를 확인하세요!
            </p>
        </div>
        """, unsafe_allow_html=True)

class ModelManager:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> CatBoostRegressor:
        try:
            model = CatBoostRegressor()
            model.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"모델 로드 오류: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        try:
            # 데이터 복사본 생성
            data_copy = data.copy()
            
            # 데이터 타입 변환
            for col in data_copy.columns:
                if col.startswith('TRAVEL_STYL_'):
                    data_copy[col] = data_copy[col].astype(int)
                elif col in NUMERICAL_FEATURES:
                    data_copy[col] = data_copy[col].astype(float)
                else:
                    data_copy[col] = data_copy[col].astype(str)
            
            # 컬럼 순서 일치시키기
            all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
            missing_cols = set(all_features) - set(data_copy.columns)
            if missing_cols:
                st.warning(f"누락된 특성: {missing_cols}")
                for col in missing_cols:
                    if col in NUMERICAL_FEATURES:
                        data_copy[col] = 0
                    else:
                        data_copy[col] = '0'
            
            # 필요한 컬럼만 선택하여 Pool 생성
            data_for_pool = data_copy[all_features].copy()
            
            # 데이터 타입 확인 및 디버깅
            for col in CATEGORICAL_FEATURES:
                if col.startswith('TRAVEL_STYL_'):
                    data_for_pool[col] = data_for_pool[col].astype(int)
                else:
                    data_for_pool[col] = data_for_pool[col].astype(str)
            
            for col in NUMERICAL_FEATURES:
                data_for_pool[col] = data_for_pool[col].astype(float)
            
            # Pool 생성 및 예측
            pool = Pool(
                data=data_for_pool,
                cat_features=[col for col in CATEGORICAL_FEATURES if not col.startswith('TRAVEL_STYL_')]
            )
            
            return self.model.predict(pool)
                
        except Exception as e:
            st.error(f"예측 오류: {e}")
            return np.array([3.0])  # 기본값 반환

class DataPreprocessor:
    @staticmethod
    def prepare_input_data(input_dict: Dict, area: str) -> pd.DataFrame:
        try:
            # 입력 데이터 복사
            input_data = input_dict.copy()
            input_data['VISIT_AREA_NM'] = area
            
            # DataFrame 생성
            df = pd.DataFrame([input_data])
            
            # 타입 변환
            for col in df.columns:
                if col.startswith('TRAVEL_STYL_'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)
                elif col in NUMERICAL_FEATURES:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                else:
                    df[col] = df[col].astype(str)
            
            return df
            
        except Exception as e:
            st.error(f"데이터 전처리 오류: {e}")
            # 기본 데이터 반환
            default_data = {
                'GENDER': '남',
                'AGE_GRP': 20.0,
                'TRAVEL_STYL_1': 4,
                'TRAVEL_STYL_2': 4,
                'TRAVEL_STYL_3': 4,
                'TRAVEL_STYL_4': 4,
                'TRAVEL_STYL_5': 4,
                'TRAVEL_STYL_6': 4,
                'TRAVEL_STYL_7': 4,
                'TRAVEL_STYL_8': 4,
                'TRAVEL_MOTIVE_1': '8',
                'TRAVEL_COMPANIONS_NUM': 0.0,
                'TRAVEL_MISSION_INT': '3',
                'VISIT_AREA_NM': area
            }
            return pd.DataFrame([default_data])

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

class TravelerInputForm:
    @staticmethod
    def get_input() -> Dict:
        with st.sidebar:
            st.header('여행자 정보 입력')
            
            # 성별 선택
            gender = st.selectbox('성별', ['남', '여'], key="gender_select")
            
            # 연령대 선택
            age_grp = st.number_input('연령대', min_value=20, max_value=70, value=20, step=10, key="age_input")
            
            # 여행 스타일 선택
            st.subheader('여행 스타일')
            travel_styl_values = TravelerInputForm._get_travel_style_inputs()
            
            # 여행 동기 선택
            travel_motive = st.selectbox(
                '주요 여행 동기', 
                options=list(TRAVEL_MOTIVE_MAPPING.keys()),
                format_func=lambda x: TRAVEL_MOTIVE_MAPPING.get(x, x),
                index=7,
                help="\n".join([f"{k}: {v}" for k, v in TRAVEL_MOTIVE_MAPPING.items()]),
                key="travel_motive_select"
            )

            # 동반자 수 입력
            companions_num = st.number_input('동반자 수', min_value=0, max_value=10, value=0, step=1, key="companions_input")
            
            # 여행 목적 선택
            travel_mission = st.selectbox(
                '여행 목적', 
                options=list(TRAVEL_MISSION_MAPPING.keys()),
                format_func=lambda x: TRAVEL_MISSION_MAPPING.get(x, x),
                index=2,
                help="\n".join([f"{k}: {v}" for k, v in TRAVEL_MISSION_MAPPING.items()]),
                key="travel_mission_select"
            )
            
            # 도움말
            with st.expander("💡 여행 스타일 설명"):
                for style, (left, right) in TRAVEL_STYLE_MAPPING.items():
                    st.markdown(f"""
                    **{style}**
                    - 1-3: {left} 선호
                    - 4: 중립
                    - 5-7: {right} 선호
                    """)
            
            return {
                'GENDER': gender,
                'AGE_GRP': float(age_grp),
                **travel_styl_values,
                'TRAVEL_MOTIVE_1': travel_motive,
                'TRAVEL_COMPANIONS_NUM': float(companions_num),
                'TRAVEL_MISSION_INT': travel_mission
            }
    
    @staticmethod
    def _get_travel_style_inputs() -> Dict[str, int]:
        travel_styl_values = {}
        for key, (left_label, right_label) in TRAVEL_STYLE_MAPPING.items():
            value = st.select_slider(
                f'{left_label} vs {right_label}',
                options=[1, 2, 3, 4, 5, 6, 7],
                value=4,
                help=f"""
                1-3: {left_label} 선호 (1: 매우 강함, 2: 강함, 3: 약간)
                4: 중립
                5-7: {right_label} 선호 (5: 약간, 6: 강함, 7: 매우 강함)
                """,
                key=f"travel_style_{key}"
            )
            travel_styl_values[key] = value
        return travel_styl_values

class ResultVisualizer:
    @staticmethod
    def display_results(results: pd.DataFrame):
        if len(results) > 30:
            results = results.head(30)
            
        st.write("### 추천 결과")
        fig = px.bar(
            results,
            x='VISIT_AREA_NM',
            y='예측 만족도',
            title='지역별 예측 만족도',
            labels={'VISIT_AREA_NM': '방문 지역', '예측 만족도': '예측 만족도 (1-5)'}
        )
        st.plotly_chart(fig)
        
        # 상세 결과 표시
        st.write("### 상세 추천 목록")
        st.dataframe(results)

def main():
    st.title('여행 만족도 예측 시스템')
    
    try:
        # 데이터 로드
        df = pd.read_csv('./data/df_filter.csv')
        
        # 모델 로드
        model_manager = ModelManager('catboost_model.cbm')
        
        # 여행 스타일 분석 결과 로드
        try:
            style_analysis = pd.read_csv('travel_style_analysis.csv')
            with st.expander("📊 여행 스타일 분석 결과"):
                st.dataframe(style_analysis)
                
            # 분포 그래프 표시
            if st.checkbox("여행 스타일 분포 보기"):
                st.image('travel_style_distribution.png')
        except Exception as e:
            st.warning(f"여행 스타일 분석 결과를 찾을 수 없습니다: {e}")
        
        # 사용자 입력 받기
        user_input = TravelerInputForm.get_input()
        
        # 모든 지역에 대해 예측 (최대 50개 지역으로 제한)
        unique_areas = df['VISIT_AREA_NM'].unique()[:50]
        predictions = []
        
        for area in unique_areas:
            # 데이터 전처리
            input_df = DataPreprocessor.prepare_input_data(user_input, area)
            
            # 예측
            try:
                pred = model_manager.predict(input_df)[0]
                predictions.append({
                    'VISIT_AREA_NM': area,
                    '예측 만족도': round(pred, 2)
                })
            except Exception as e:
                st.error(f"{area} 지역 예측 오류: {e}")
                continue
        
        # 결과를 데이터프레임으로 변환하고 정렬
        if predictions:
            results_df = pd.DataFrame(predictions)
            results_df = results_df.sort_values('예측 만족도', ascending=False)
            
            # 결과 시각화
            ResultVisualizer.display_results(results_df)
        else:
            st.warning("예측 결과가 없습니다.")
        
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")

if __name__ == '__main__':
    main()

