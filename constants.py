"""
여행 만족도 예측 시스템에서 사용하는 상수 및 매핑 정보
"""

# 모델 특성 정의
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

# 파일 경로
DATA_PATH = './data/df_filter.csv'
MODEL_PATH = 'catboost_model.cbm'
STYLE_ANALYSIS_PATH = 'travel_style_analysis.csv'
STYLE_DISTRIBUTION_IMG = 'travel_style_distribution.png'

# 기본 사용자 정보
DEFAULT_USER_INPUT = {
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
    'TRAVEL_MISSION_INT': '3'
}

# 만족도 등급 매핑
SATISFACTION_LEVELS = {
    (4.5, 5.0): {"등급": "최상", "설명": "매우 높은 만족도가 예상됩니다. 꼭 방문해보세요!"},
    (4.0, 4.5): {"등급": "상", "설명": "높은 만족도가 예상됩니다. 추천합니다."},
    (3.5, 4.0): {"등급": "중상", "설명": "평균 이상의 만족도가 예상됩니다."},
    (3.0, 3.5): {"등급": "중", "설명": "보통 수준의 만족도가 예상됩니다."},
    (2.5, 3.0): {"등급": "중하", "설명": "평균 이하의 만족도가 예상됩니다."},
    (2.0, 2.5): {"등급": "하", "설명": "낮은 만족도가 예상됩니다."},
    (1.0, 2.0): {"등급": "최하", "설명": "매우 낮은 만족도가 예상됩니다. 다른 지역을 고려해보세요."}
} 