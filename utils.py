"""
여행 만족도 예측 시스템에서 사용하는 유틸리티 함수
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import matplotlib.font_manager as fm
import streamlit as st
import base64
from PIL import Image
import io
from typing import Dict, Tuple, List, Any, Optional

# 한글 폰트 설정
def set_korean_font():
    """
    matplotlib에서 한글 폰트를 사용할 수 있도록 설정하는 함수입니다.
    운영체제에 따라 적절한 폰트를 찾아 설정합니다.
    """
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

def get_satisfaction_level(score: float) -> Dict[str, str]:
    """
    만족도 점수에 따른 등급 및 설명을 반환합니다.
    
    Args:
        score: 만족도 점수 (1-5 사이)
        
    Returns:
        등급 및 설명 정보를 담은 딕셔너리
    """
    from constants import SATISFACTION_LEVELS
    
    for (min_score, max_score), level_info in SATISFACTION_LEVELS.items():
        if min_score <= score <= max_score:
            return level_info
    
    # 기본값 반환
    return {"등급": "알 수 없음", "설명": "만족도 정보를 확인할 수 없습니다."}

def create_color_scale(scores: List[float]) -> List[str]:
    """
    만족도 점수에 따라 색상 스케일을 생성합니다.
    
    Args:
        scores: 만족도 점수 리스트
        
    Returns:
        색상 코드 리스트
    """
    colors = []
    for score in scores:
        if score >= 4.5:
            colors.append('#1a9641')  # 매우 높음 (짙은 녹색)
        elif score >= 4.0:
            colors.append('#51b364')  # 높음 (녹색)
        elif score >= 3.5:
            colors.append('#8ace7e')  # 평균 이상 (연한 녹색)
        elif score >= 3.0:
            colors.append('#d2e6a3')  # 평균 (연두색)
        elif score >= 2.5:
            colors.append('#fee08b')  # 평균 이하 (노란색)
        elif score >= 2.0:
            colors.append('#fc9272')  # 낮음 (연한 빨간색)
        else:
            colors.append('#e31a1c')  # 매우 낮음 (짙은 빨간색)
    return colors

def get_binary_image(image_path: str) -> Optional[str]:
    """
    이미지 파일을 base64 인코딩된 문자열로 변환합니다.
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        base64 인코딩된 이미지 문자열 또는 None (파일이 없는 경우)
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"이미지 로드 오류: {str(e)}")
        return None

def display_user_profile(user_input: Dict[str, Any]) -> None:
    """
    사용자 입력 정보를 요약하여 표시합니다.
    
    Args:
        user_input: 사용자 입력 정보를 담은 딕셔너리
    """
    from constants import (
        AGE_GROUP_MAPPING, TRAVEL_STYLE_MAPPING, 
        TRAVEL_MOTIVE_MAPPING, TRAVEL_MISSION_MAPPING
    )
    
    # 기본 정보
    st.markdown("#### 🧑 기본 정보")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**성별**: {user_input['GENDER']}")
    with col2:
        age = int(user_input['AGE_GRP'])
        st.markdown(f"**연령대**: {AGE_GROUP_MAPPING.get(age, f'{age}대')}")
    with col3:
        st.markdown(f"**동반자 수**: {int(user_input['TRAVEL_COMPANIONS_NUM'])}명")
    
    # 여행 동기와 목적
    st.markdown("#### 🎯 여행 동기 및 목적")
    col1, col2 = st.columns(2)
    with col1:
        motive_key = user_input['TRAVEL_MOTIVE_1']
        st.markdown(f"**주요 동기**: {TRAVEL_MOTIVE_MAPPING.get(motive_key, '알 수 없음')}")
    with col2:
        mission_key = user_input['TRAVEL_MISSION_INT']
        st.markdown(f"**여행 목적**: {TRAVEL_MISSION_MAPPING.get(mission_key, '알 수 없음')}")
    
    # 여행 스타일 레이더 차트
    st.markdown("#### 🧭 여행 스타일 프로필")
    travel_styles = {key: value for key, value in user_input.items() if key.startswith('TRAVEL_STYL_')}
    
    # 레이더 차트용 데이터 준비
    set_korean_font()
    categories = [f"{left} vs {right}" for key, (left, right) in TRAVEL_STYLE_MAPPING.items()]
    values = list(travel_styles.values())
    
    # 레이더 차트 그리기
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    
    # 각도 계산
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 닫힌 다각형을 만들기 위해
    
    # 값 준비 (1-7 스케일을 0-1로 변환)
    values_normalized = [(v - 1) / 6 for v in values]
    values_normalized += values_normalized[:1]  # 닫힌 다각형을 만들기 위해
    
    # 그래프 그리기
    ax.plot(angles, values_normalized, linewidth=2, linestyle='solid', label="사용자 스타일")
    ax.fill(angles, values_normalized, alpha=0.25)
    
    # 중립 값 표시
    neutral = [(4 - 1) / 6] * len(categories)
    neutral += neutral[:1]
    ax.plot(angles, neutral, linewidth=1, linestyle='--', color='gray', alpha=0.5, label="중립")
    
    # 축 설정
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_yticks([])  # y축 눈금 제거
    
    # 범례 및 제목
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    st.pyplot(fig)
    
    # 스타일 해석
    st.markdown("##### 선호 스타일 해석")
    for key, value in travel_styles.items():
        idx = int(key.split('_')[-1])
        style_key = f'TRAVEL_STYL_{idx}'
        left, right = TRAVEL_STYLE_MAPPING[style_key]
        
        if value < 4:
            preference = f"**{left}** 선호 (강도: {4-value})"
            emoji = "🔵"
        elif value > 4:
            preference = f"**{right}** 선호 (강도: {value-4})"
            emoji = "🔴"
        else:
            preference = "중립"
            emoji = "⚪"
            
        st.markdown(f"{emoji} {left} vs {right}: {preference}") 