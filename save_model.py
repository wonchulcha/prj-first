import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.font_manager as fm
import platform

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
                    print(f"한글 폰트 설정 완료: {font_name}")
                    return
            except:
                continue
        print("사용 가능한 한글 폰트를 찾을 수 없습니다.")
    else:
        # macOS 또는 Linux의 경우
        try:
            plt.rc('font', family='AppleGothic' if system_name == 'Darwin' else 'NanumGothic')
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            print(f"{system_name} 시스템용 한글 폰트 설정 완료")
        except:
            print(f"{system_name} 시스템에서 한글 폰트 설정에 실패했습니다.")

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

def load_data(file_path):
    """데이터 파일을 로드합니다."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        raise

def analyze_travel_styles(df):
    """여행 스타일 데이터를 분석하고 시각화합니다."""
    style_columns = [f'TRAVEL_STYL_{i}' for i in range(1, 9)]
    results = []
    
    # 한글 폰트 설정
    set_korean_font()
    
    plt.figure(figsize=(15, 10))
    for idx, style in enumerate(style_columns, 1):
        # 기본 통계량 계산
        stats = df[style].describe()
        
        # 선호도 분포 계산
        value_counts = df[style].value_counts().sort_index()
        
        # 양극성 분석
        left_preference = (value_counts[1:4].sum() / len(df)) * 100  # 1-3: 왼쪽 성향
        neutral = (value_counts[4] / len(df)) * 100  # 4: 중립
        right_preference = (value_counts[5:8].sum() / len(df)) * 100  # 5-7: 오른쪽 성향
        
        left_label, right_label = TRAVEL_STYLE_MAPPING[style]
        
        # 평균값이 4보다 작으면 왼쪽, 크면 오른쪽 선호
        mean_preference = stats['mean']
        preference_direction = left_label if mean_preference < 4 else right_label
        
        results.append({
            '스타일': style,
            '왼쪽 성향': left_label,
            '오른쪽 성향': right_label,
            '평균': mean_preference,
            '중앙값': stats['50%'],
            '표준편차': stats['std'],
            f'{left_label} 선호(1-3)': f"{left_preference:.1f}%",
            '중립(4)': f"{neutral:.1f}%",
            f'{right_label} 선호(5-7)': f"{right_preference:.1f}%",
            '주요 성향': preference_direction
        })
        
        # 분포 시각화
        plt.subplot(4, 2, idx)
        sns.histplot(data=df, x=style, bins=7)
        plt.title(f'{left_label} vs {right_label}')
        plt.xlabel('1(왼쪽) ← 4(중립) → 7(오른쪽)')
    
    plt.tight_layout()
    plt.savefig('travel_style_distribution.png', dpi=300)
    plt.close()
    
    return pd.DataFrame(results)

def main():
    try:
        print("데이터 로드 중...")
        # 데이터 로드
        df_place = load_data('./data/tn_visit_area_info.csv')
        df_travel = load_data('./data/tn_travel.csv')
        df_traveler = load_data('./data/tn_traveller_master.csv')

        print("데이터 병합 중...")
        # 데이터 병합
        df = pd.merge(df_place, df_travel, on='TRAVEL_ID', how='left')
        df = pd.merge(df, df_traveler, on='TRAVELER_ID', how='left')

        print("데이터 전처리 중...")
        # 데이터 전처리
        df_filter = df[~df['TRAVEL_MISSION_CHECK'].isnull()].copy()
        df_filter.loc[:, 'TRAVEL_MISSION_INT'] = df_filter['TRAVEL_MISSION_CHECK'].str.split(';').str[0].astype(int)

        # 필요한 컬럼만 선택
        df_filter = df_filter[[
            'GENDER',
            'AGE_GRP',
            'TRAVEL_STYL_1',
            'TRAVEL_STYL_2',
            'TRAVEL_STYL_3',
            'TRAVEL_STYL_4',
            'TRAVEL_STYL_5',
            'TRAVEL_STYL_6',
            'TRAVEL_STYL_7',
            'TRAVEL_STYL_8',
            'TRAVEL_MOTIVE_1',
            'TRAVEL_MISSION_INT',
            'VISIT_AREA_NM',
            'DGSTFN'
        ]]

        # 결측치 제거
        df_filter = df_filter.dropna()

        # 데이터 타입 변환
        print("데이터 타입 변환 중...")
        # 여행 스타일 컬럼 정수형으로 변환
        style_cols = [col for col in df_filter.columns if col.startswith('TRAVEL_STYL_')]
        for col in style_cols:
            df_filter[col] = df_filter[col].astype(int)
            
        # 범주형 변수는 문자열로 변환 (여행 스타일 제외)
        categorical_cols = ['GENDER', 'TRAVEL_MOTIVE_1', 'TRAVEL_MISSION_INT', 'VISIT_AREA_NM']
        for col in categorical_cols:
            df_filter[col] = df_filter[col].astype(str)
        
        # 수치형 변수는 실수형으로 변환
        numerical_cols = ['AGE_GRP', 'DGSTFN']
        for col in numerical_cols:
            df_filter[col] = df_filter[col].astype(float)

        print("여행 스타일 분석 중...")
        # 여행 스타일 분석 수행
        style_analysis = analyze_travel_styles(df_filter)
        print("\n=== 여행 스타일 분석 결과 ===")
        print(style_analysis.to_string())
        
        # 범주형 변수 정의
        categorical_features = [
            'GENDER',
            'TRAVEL_STYL_1',
            'TRAVEL_STYL_2',
            'TRAVEL_STYL_3',
            'TRAVEL_STYL_4',
            'TRAVEL_STYL_5',
            'TRAVEL_STYL_6',
            'TRAVEL_STYL_7',
            'TRAVEL_STYL_8',
            'TRAVEL_MOTIVE_1',
            'TRAVEL_MISSION_INT',
            'VISIT_AREA_NM'
        ]

        print("데이터 분할 중...")
        # 학습/테스트 데이터 분할
        train_df, test_df = train_test_split(df_filter, test_size=0.2, random_state=42)

        print("CatBoost 데이터셋 생성 중...")
        # CatBoost 데이터셋 생성
        train_pool = Pool(
            train_df.drop('DGSTFN', axis=1),
            train_df['DGSTFN'],
            cat_features=[col for col in categorical_features if not col.startswith('TRAVEL_STYL_')]
        )

        test_pool = Pool(
            test_df.drop('DGSTFN', axis=1),
            test_df['DGSTFN'],
            cat_features=[col for col in categorical_features if not col.startswith('TRAVEL_STYL_')]
        )

        print("모델 학습 중...")
        # 모델 학습
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=100
        )

        model.fit(train_pool, eval_set=test_pool)

        print("모델 평가 중...")
        # 모델 성능 평가
        predictions = model.predict(test_pool)
        rmse = np.sqrt(mean_squared_error(test_df['DGSTFN'], predictions))
        mae = mean_absolute_error(test_df['DGSTFN'], predictions)
        r2 = r2_score(test_df['DGSTFN'], predictions)

        print("\n=== 모델 성능 ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"테스트 데이터 크기: {len(test_df):,}")

        print("데이터 및 모델 저장 중...")
        # 데이터와 모델 저장
        df_filter.to_csv('./data/df_filter.csv', encoding='utf-8', index=False)
        model.save_model('catboost_model.cbm')
        
        # 분석 결과 저장
        style_analysis.to_csv('travel_style_analysis.csv', encoding='utf-8', index=False)
        
        print("모든 과정이 완료되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        raise

if __name__ == '__main__':
    main()
