import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_traveler_input():
    with st.sidebar:
        st.header('여행자 정보 입력')
        
        # 성별 선택
        gender = st.selectbox('성별', ['남', '여'])
        
        # 연령대 선택
        age_grp = st.number_input('연령대', min_value=20.0, max_value=70.0, value=20.0, step=10.0)
        
        # 여행 스타일 선택
        st.subheader('여행 스타일')
        travel_styl = {
            'TRAVEL_STYL_1': '자연/풍경 관광',
            'TRAVEL_STYL_2': '역사/문화 관광',
            'TRAVEL_STYL_3': '음식/맛집 탐방',
            'TRAVEL_STYL_4': '쇼핑/레저',
            'TRAVEL_STYL_5': '휴양/힐링',
            'TRAVEL_STYL_6': '체험/액티비티',
            'TRAVEL_STYL_7': '예술/공연',
            'TRAVEL_STYL_8': '교육/학습'
        }
        
        travel_styl_values = {}
        for key, desc in travel_styl.items():
            value = st.selectbox(f'{desc} (1-5)', 
                               options=['1', '2', '3', '4', '5'],
                               index=0,
                               help='1: 매우 낮음, 2: 낮음, 3: 보통, 4: 높음, 5: 매우 높음')
            travel_styl_values[key] = value
        
        # 여행 동기 선택
        travel_motive = st.selectbox('주요 여행 동기', 
                                   options=['1', '2', '3', '4', '5', '6', '7', '8'],
                                   index=7,
                                   help='1: 가족여행, 2: 친구/연인과의 여행, 3: 업무/출장, 4: 교육/연수, 5: 건강/휴양, 6: 취미/레저, 7: 종교/순례, 8: 기타')
        
        # 동반자 수 입력
        companions_num = st.number_input('동반자 수', min_value=0.0, max_value=10.0, value=0.0, step=1.0)
        
        # 여행 목적 선택
        travel_mission = st.selectbox('여행 목적', 
                                    options=['1', '2', '3', '4', '5'],
                                    index=2,
                                    help='1: 휴식/힐링, 2: 문화체험, 3: 관광/명소방문, 4: 음식/맛집, 5: 쇼핑/레저')
        
        # 입력값을 딕셔너리로 반환
        return {
            'GENDER': gender,
            'AGE_GRP': age_grp,
            **travel_styl_values,
            'TRAVEL_MOTIVE_1': travel_motive,
            'TRAVEL_COMPANIONS_NUM': companions_num,
            'TRAVEL_MISSION_INT': travel_mission
        }

def main():
    
    # 데이터 로드
    df_filter = pd.read_csv('df_filter.csv')
    
    # 모델 로드
    loaded_model = CatBoostRegressor()
    loaded_model.load_model('catboost_model.cbm')
    
    st.title('여행지 추천 시스템')
    
    # 모델 성능 지표 표시
    st.subheader('모델 성능 지표')
    
    # 테스트 데이터 준비
    test_data = df_filter.sample(frac=0.2, random_state=42)
    train_data = df_filter.drop(test_data.index)
    
    # 범주형 특성 정의
    cat_features = [
        'GENDER', 
        'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 
        'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 
        'TRAVEL_STYL_7', 'TRAVEL_STYL_8', 
        'TRAVEL_MOTIVE_1', 
        'TRAVEL_MISSION_INT', 
        'VISIT_AREA_NM'
    ]
    
    # 테스트 데이터에 대한 예측
    test_pool = Pool(test_data.drop('DGSTFN', axis=1), cat_features=cat_features)
    test_predictions = loaded_model.predict(test_pool)
    
    # 성능 지표 계산
    mse = mean_squared_error(test_data['DGSTFN'], test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data['DGSTFN'], test_predictions)
    r2 = r2_score(test_data['DGSTFN'], test_predictions)
    
    # 성능 지표 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{rmse:.4f}")
    with col2:
        st.metric("MAE", f"{mae:.4f}")
    with col3:
        st.metric("R² Score", f"{r2:.4f}")
    with col4:
        st.metric("테스트 데이터 크기", f"{len(test_data):,}")
    
    # 사이드바에서 여행자 정보 입력 받기
    traveler = get_traveler_input()
    
    # 여행지 추천
    area_names = df_filter[['VISIT_AREA_NM']].drop_duplicates()
    results = pd.DataFrame([], columns=['AREA', 'SCORE'])
    result_list = []
    
    for area in area_names['VISIT_AREA_NM']:
        input_data = traveler.copy()
        input_data['VISIT_AREA_NM'] = area
        
        # 입력 데이터를 DataFrame으로 변환
        input_df = pd.DataFrame([input_data])
        
        # 데이터 타입 변환
        for col in input_df.columns:
            if col in ['AGE_GRP', 'TRAVEL_COMPANIONS_NUM']:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            elif col in cat_features:
                input_df[col] = input_df[col].astype(str)
        
        # 결측치 처리
        input_df = input_df.fillna(0)
        
        # CatBoost Pool 생성 및 예측
        predict_pool = Pool(input_df, cat_features=cat_features)
        score = loaded_model.predict(predict_pool)[0]
        result_list.append([area, score])
    
    results = pd.DataFrame(result_list, columns=['AREA', 'SCORE'])
    
    # 상위 10개 여행지
    st.subheader('추천 여행지 TOP 10')
    top_10 = results.sort_values('SCORE', ascending=False)[:10]
    st.dataframe(top_10, use_container_width=True)
    
    # 하위 10개 여행지
    st.subheader('비추천 여행지 TOP 10')
    bottom_10 = results.sort_values('SCORE', ascending=True)[:10]
    st.dataframe(bottom_10, use_container_width=True)
    
    # 시각화
    st.subheader('여행지 추천 점수 분포')
    fig = px.bar(results.sort_values('SCORE', ascending=False), 
                 x='AREA', y='SCORE',
                 title='여행지별 추천 점수')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()






















