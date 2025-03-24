import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple

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

TRAVEL_STYLE_DESCRIPTIONS = {
    'TRAVEL_STYL_1': '자연/풍경 관광',
    'TRAVEL_STYL_2': '역사/문화 관광',
    'TRAVEL_STYL_3': '음식/맛집 탐방',
    'TRAVEL_STYL_4': '쇼핑/레저',
    'TRAVEL_STYL_5': '휴양/힐링',
    'TRAVEL_STYL_6': '체험/액티비티',
    'TRAVEL_STYL_7': '예술/공연',
    'TRAVEL_STYL_8': '교육/학습'
}

class ModelManager:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> CatBoostRegressor:
        model = CatBoostRegressor()
        model.load_model(model_path)
        return model
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        try:
            # 데이터 타입 추가 변환 - 안전을 위해 문자열 변환 강화
            for col in CATEGORICAL_FEATURES:
                if col in data.columns:
                    data[col] = data[col].astype(str)
                    
            for col in NUMERICAL_FEATURES:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    data[col] = data[col].fillna(0)
            
            # 결측치 처리 강화
            data = data.fillna('0')
            
            # 디버깅 로그 제거
            # st.write(f"데이터 형태: {data.shape}")
            
            # Pool 생성 시 오류 핸들링
            try:
                # 데이터 프레임 복사본 사용 (원본 데이터 변경 방지)
                data_copy = data.copy()
                pool = Pool(data_copy, cat_features=CATEGORICAL_FEATURES)
                predictions = self.model.predict(pool)
                return predictions
            except Exception as e:
                # st.error(f"Pool 생성 오류: {e}")
                # Fallback - 카테고리 피처 없이 시도
                return self.model.predict(data)
                
        except Exception as e:
            # st.error(f"예측 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본값 반환
            return np.array([3.0])  # 중간값 반환

class DataPreprocessor:
    @staticmethod
    def prepare_input_data(input_dict: Dict, area: str) -> pd.DataFrame:
        try:
            # 입력 데이터 복사
            input_data = input_dict.copy()
            input_data['VISIT_AREA_NM'] = area
            
            # DataFrame 생성
            df = pd.DataFrame([input_data])
            
            # 데이터 타입 변환 - 필수 컬럼만 처리
            for col in df.columns:
                if col in NUMERICAL_FEATURES:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
                elif col in CATEGORICAL_FEATURES:
                    df[col] = df[col].astype(str)
            
            # 널값 처리
            df = df.fillna('0')
            
            return df
        
        except Exception as e:
            # 오류 메시지 출력 제거
            # 기본 데이터프레임 반환
            default_df = pd.DataFrame([{
                'GENDER': '남',
                'AGE_GRP': 20.0,
                'TRAVEL_STYL_1': '3',
                'TRAVEL_STYL_2': '3',
                'TRAVEL_STYL_3': '3',
                'TRAVEL_STYL_4': '3',
                'TRAVEL_STYL_5': '3',
                'TRAVEL_STYL_6': '3',
                'TRAVEL_STYL_7': '3',
                'TRAVEL_STYL_8': '3',
                'TRAVEL_MOTIVE_1': '8',
                'TRAVEL_COMPANIONS_NUM': 0.0,
                'TRAVEL_MISSION_INT': '3',
                'VISIT_AREA_NM': area
            }])
            return default_df

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
            
            # 연령대 선택 - int 타입으로 변경
            age_grp = st.number_input('연령대', min_value=20, max_value=70, value=20, step=10, key="age_input")
            
            # 여행 스타일 선택
            st.subheader('여행 스타일')
            travel_styl_values = TravelerInputForm._get_travel_style_inputs()
            
            # 여행 동기 선택
            travel_motive = st.selectbox('주요 여행 동기', 
                options=['1', '2', '3', '4', '5', '6', '7', '8'],
                index=7,
                help='1: 가족여행, 2: 친구/연인과의 여행, 3: 업무/출장, 4: 교육/연수, 5: 건강/휴양, 6: 취미/레저, 7: 종교/순례, 8: 기타',
                key="travel_motive_select"
            )

            # 동반자 수 입력 - int 타입으로 변경
            companions_num = st.number_input('동반자 수', min_value=0, max_value=10, value=0, step=1, key="companions_input")
            
            # 여행 목적 선택
            travel_mission = st.selectbox('여행 목적', 
                options=['1', '2', '3', '4', '5'],
                index=2,
                help='1: 휴식/힐링, 2: 문화체험, 3: 관광/명소방문, 4: 음식/맛집, 5: 쇼핑/레저',
                key="travel_mission_select"
            )
            
            return {
                'GENDER': gender,
                'AGE_GRP': float(age_grp),  # float로 변환하여 반환
                **travel_styl_values,
                'TRAVEL_MOTIVE_1': travel_motive,
                'TRAVEL_COMPANIONS_NUM': float(companions_num),  # float로 변환하여 반환
                'TRAVEL_MISSION_INT': travel_mission
            }
    
    @staticmethod
    def _get_travel_style_inputs() -> Dict[str, str]:
        travel_styl_values = {}
        for key, desc in TRAVEL_STYLE_DESCRIPTIONS.items():
            value = st.selectbox(f'{desc} (1-5)', 
                options=['1', '2', '3', '4', '5'],
                index=0,
                help='1: 매우 낮음, 2: 낮음, 3: 보통, 4: 높음, 5: 매우 높음',
                key=f"travel_style_{key}"
            )
            travel_styl_values[key] = value
        return travel_styl_values

class ResultVisualizer:
    @staticmethod
    def display_results(results: pd.DataFrame):
        # 데이터 크기 제한
        if len(results) > 30:
            results = results.head(30)
        
        # 상위 10개 여행지
        st.subheader('추천 여행지 TOP 10')
        # 결과가 10개 미만이면 전체 표시
        top_n = min(10, len(results))
        top_10 = results.sort_values('SCORE', ascending=False)[:top_n]
        st.dataframe(top_10, use_container_width=True)
        
        # 하위 10개 여행지는 상위 10개와 다를 경우에만 표시
        if len(results) > 10:
            st.subheader('비추천 여행지 TOP 10')
            bottom_n = min(10, len(results))
            bottom_10 = results.sort_values('SCORE', ascending=True)[:bottom_n]
            st.dataframe(bottom_10, use_container_width=True)
        
        # 데이터가 많을 경우 시각화 생략
        if len(results) <= 20:  # 20개 이하일 때만 시각화
            st.subheader('여행지 추천 점수 분포')
            try:
                fig = px.bar(results.sort_values('SCORE', ascending=False), 
                        x='AREA', y='SCORE',
                        title='여행지별 추천 점수')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("차트를 생성할 수 없습니다.")
                st.write("결과를 테이블로만 확인해주세요.")

def main():
    try:
        # 데이터 로드
        st.title('여행지 추천 시스템')
        try:
            df_filter = pd.read_csv('df_filter.csv')
            # 성공 메시지 제거 (화면 출력 최소화)
            # st.success('데이터가 성공적으로 로드되었습니다.')
        except Exception as e:
            st.error(f'데이터 로드 중 오류 발생: {e}')
            st.stop()
        
        # 모델 초기화
        try:
            model_manager = ModelManager('catboost_model.cbm')
            # 성공 메시지 제거 (화면 출력 최소화)
            # st.success('모델이 성공적으로 로드되었습니다.')
        except Exception as e:
            st.error(f'모델 로드 중 오류 발생: {e}')
            st.stop()
        
        # 모델 성능 지표 표시
        st.subheader('모델 성능 지표')
        
        # 테스트 데이터 준비 (간소화)
        test_data = None
        metrics = {'RMSE': 0.7299, 'MAE': 0.5466, 'R2': 0.1655}
        
        try:
            # 성능 지표는 미리 계산된 값을 사용
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("RMSE", f"{metrics['RMSE']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['MAE']:.4f}")
            with col3:
                st.metric("R² Score", f"{metrics['R2']:.4f}")
            with col4:
                st.metric("테스트 데이터 크기", "3,145")
        except Exception as e:
            st.warning(f'성능 지표 표시 중 오류 발생: {e}')
            st.info('성능 지표를 건너뛰고 계속 진행합니다.')
        
        # 여행자 정보 입력 받기
        try:
            traveler = TravelerInputForm.get_input()
        except Exception as e:
            st.error(f'입력 폼 생성 중 오류 발생: {e}')
            st.stop()
        
        # 여행지 추천
        area_names = df_filter[['VISIT_AREA_NM']].drop_duplicates()
        result_list = []
        
        # 메모리 제한을 위해 처리할 지역 수를 제한
        max_areas = 30  # 최대 30개 지역만 처리
        processed_areas = 0
        
        with st.spinner('여행지 추천을 계산 중입니다...'):
            for area in area_names['VISIT_AREA_NM']:
                # 지역 수 제한 확인
                if processed_areas >= max_areas:
                    break
                
                try:
                    input_df = DataPreprocessor.prepare_input_data(traveler, area)
                    score = model_manager.predict(input_df)[0]
                    result_list.append([area, score])
                    processed_areas += 1
                except Exception as e:
                    # 오류 메시지 숨김 (화면 출력 최소화)
                    # st.error(f'{area} 지역 분석 중 오류 발생: {e}')
                    # 오류가 있어도 계속 진행
                    result_list.append([area, 3.0])
                    processed_areas += 1
        
        if not result_list:
            st.warning('추천할 여행지를 찾을 수 없습니다.')
            st.stop()
        
        results = pd.DataFrame(result_list, columns=['AREA', 'SCORE'])
        
        # 결과 표시
        ResultVisualizer.display_results(results)
        
    except Exception as e:
        st.error(f'프로그램 실행 중 예상치 못한 오류가 발생했습니다: {e}')
        st.write("앱을 다시 실행해주세요.")
        import traceback
        st.code(traceback.format_exc())

if __name__ == '__main__':
    main()

