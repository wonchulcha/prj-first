"""
여행 만족도 예측 시스템의 UI 컴포넌트
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional
from constants import (
    TRAVEL_STYLE_MAPPING, TRAVEL_MOTIVE_MAPPING, 
    TRAVEL_MISSION_MAPPING, AGE_GROUP_MAPPING,
    STYLE_ANALYSIS_PATH, STYLE_DISTRIBUTION_IMG
)
from utils import set_korean_font, get_satisfaction_level, create_color_scale, display_user_profile

class AppStyles:
    """
    앱의 스타일 및 CSS를 관리하는 클래스
    """
    @staticmethod
    def apply_styles():
        """스타일시트 적용"""
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
            
            /* 카드 그리드 */
            .recommendation-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            
            .recommendation-card {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                padding: 15px;
                transition: transform 0.3s ease;
            }
            
            .recommendation-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }
            
            .card-title {
                font-size: 1.2rem;
                font-weight: bold;
                margin-bottom: 10px;
                color: #1E88E5;
            }
            
            .metric {
                font-size: 1.8rem;
                font-weight: bold;
                color: #26A69A;
            }
            
            .rating {
                display: flex;
                margin-top: 10px;
            }
            
            .star-filled {
                color: #FFD700;
            }
            
            .star-empty {
                color: #E0E0E0;
            }
            
            .grade-tag {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                margin-top: 10px;
            }
            
            .grade-top {
                background-color: #D4EDDA;
                color: #155724;
            }
            
            .grade-mid {
                background-color: #FFF3CD;
                color: #856404;
            }
            
            .grade-low {
                background-color: #F8D7DA;
                color: #721C24;
            }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def set_page_config():
        """페이지 설정"""
        st.set_page_config(
            page_title="여행 만족도 예측 시스템",
            page_icon="🧳",
            layout="wide",
            initial_sidebar_state="expanded",
        )


class TravelerInputForm:
    """
    여행자 정보 입력 폼을 관리하는 클래스
    """
    @staticmethod
    def get_input() -> Dict:
        """
        사용자 입력 정보 수집
        
        Returns:
            사용자 입력 정보 딕셔너리
        """
        with st.sidebar:
            st.header('여행자 정보 입력')
            
            # 성별 선택
            gender = st.selectbox('성별', ['남', '여'], key="gender_select")
            
            # 연령대 선택
            age_grp = st.number_input('연령대', min_value=20, max_value=70, value=30, step=10, key="age_input")
            
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
            companions_num = st.number_input('동반자 수', min_value=0, max_value=10, value=1, step=1, key="companions_input")
            
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
        """
        여행 스타일 입력 수집
        
        Returns:
            여행 스타일 입력 딕셔너리
        """
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
    """
    추천 결과 시각화를 담당하는 클래스
    """
    @staticmethod
    def display_results(results: pd.DataFrame):
        """
        추천 결과 시각화
        
        Args:
            results: 추천 결과 데이터프레임
        """
        if len(results) == 0:
            st.warning("추천 결과가 없습니다.")
            return
            
        st.markdown('<h2 class="sub-header">🔍 추천 여행지</h2>', unsafe_allow_html=True)
        
        # Top 3 지역 카드 표시
        top3 = results.head(3)
        col1, col2, col3 = st.columns(3)
        
        for idx, (i, row) in enumerate(top3.iterrows()):
            col = [col1, col2, col3][idx]
            
            with col:
                level_info = get_satisfaction_level(row['예측 만족도'])
                
                # 등급에 따른 배경색 결정
                if level_info['등급'] in ['최상', '상']:
                    grade_class = 'grade-top'
                elif level_info['등급'] in ['중상', '중']:
                    grade_class = 'grade-mid'
                else:
                    grade_class = 'grade-low'
                
                # 별점 계산 (1-5)
                stars = int(row['예측 만족도'])
                half_star = round(row['예측 만족도'] - stars, 1) >= 0.5
                empty_stars = 5 - stars - (1 if half_star else 0)
                
                star_html = (
                    '★' * stars + 
                    ('⯪' if half_star else '') + 
                    '☆' * empty_stars
                )
                
                st.markdown(f"""
                <div class="recommendation-card">
                    <div class="card-title">#{idx+1} {row['VISIT_AREA_NM']}</div>
                    <div class="metric">{row['예측 만족도']}/5.0</div>
                    <div class="rating">
                        <span style="font-size: 1.5rem; color: #FFD700;">{star_html}</span>
                    </div>
                    <div class="grade-tag {grade_class}">{level_info['등급']}</div>
                    <p>{level_info['설명']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 차트 표시
        st.markdown('<h3>지역별 예측 만족도</h3>', unsafe_allow_html=True)
        ResultVisualizer.plot_satisfaction_chart(results)
        
        # 상세 데이터 표시
        with st.expander("📋 상세 추천 목록"):
            st.dataframe(
                results.rename(columns={'VISIT_AREA_NM': '방문 지역'}),
                column_config={
                    "방문 지역": st.column_config.TextColumn("방문 지역"),
                    "예측 만족도": st.column_config.NumberColumn(
                        "예측 만족도",
                        help="모델이 예측한 만족도 점수 (1-5)",
                        format="%.2f"
                    )
                }
            )
    
    @staticmethod
    def plot_satisfaction_chart(results: pd.DataFrame, max_areas: int = 15):
        """
        만족도 차트 그리기
        
        Args:
            results: 추천 결과 데이터프레임
            max_areas: 차트에 표시할 최대 지역 수
        """
        # 데이터 준비
        plot_data = results.head(max_areas).copy()
        plot_data['색상'] = create_color_scale(plot_data['예측 만족도'])
        
        # 차트 생성
        fig = px.bar(
            plot_data,
            x='VISIT_AREA_NM',
            y='예측 만족도',
            color='예측 만족도',
            color_continuous_scale=['#e31a1c', '#fc9272', '#fee08b', '#d2e6a3', '#8ace7e', '#51b364', '#1a9641'],
            range_color=[1, 5],
            labels={'VISIT_AREA_NM': '방문 지역', '예측 만족도': '예측 만족도 (1-5)'},
            height=500
        )
        
        # 차트 디자인 수정
        fig.update_layout(
            xaxis_title="방문 지역",
            yaxis_title="예측 만족도 (1-5)",
            font=dict(size=14),
            plot_bgcolor='rgba(0,0,0,0.02)',
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=12),
                tickmode='array',
                tickvals=plot_data['VISIT_AREA_NM']
            ),
            yaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=0.5,
                range=[1, 5.2]
            ),
            margin=dict(l=50, r=50, b=100, t=50)
        )
        
        # 그리드 라인 추가
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black'
        )
        
        # 값 표시 추가
        for i, row in plot_data.iterrows():
            fig.add_annotation(
                x=row['VISIT_AREA_NM'],
                y=row['예측 만족도'] + 0.15,
                text=f"{row['예측 만족도']:.2f}",
                showarrow=False,
                font=dict(size=10)
            )
        
        st.plotly_chart(fig, use_container_width=True)


class IntroPage:
    """
    소개 페이지를 관리하는 클래스
    """
    @staticmethod
    def show():
        """소개 페이지 표시"""
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
                style_analysis = pd.read_csv(STYLE_ANALYSIS_PATH)
                with st.expander("📊 여행 스타일 분석 결과"):
                    st.dataframe(style_analysis)
                    
                if st.checkbox("여행 스타일 분포 보기"):
                    set_korean_font()  # 한글 폰트 설정
                    try:
                        st.image(STYLE_DISTRIBUTION_IMG)
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


class UserProfilePage:
    """
    사용자 프로필 페이지를 관리하는 클래스
    """
    @staticmethod
    def show(user_input: Dict):
        """
        사용자 프로필 페이지 표시
        
        Args:
            user_input: 사용자 입력 정보
        """
        st.markdown('<h1 class="main-header">👤 사용자 프로필</h1>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        display_user_profile(user_input)
        st.markdown('</div>', unsafe_allow_html=True) 