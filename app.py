"""
여행 만족도 예측 시스템 메인 애플리케이션

실행 방법: streamlit run app.py
"""

import streamlit as st
from constants import STYLE_ANALYSIS_PATH, STYLE_DISTRIBUTION_IMG
from model import TravelRecommender
from ui import AppStyles, TravelerInputForm, ResultVisualizer, IntroPage, UserProfilePage
from utils import set_korean_font

def main():
    """메인 애플리케이션 실행"""
    # 앱 스타일 및 페이지 설정
    AppStyles.set_page_config()
    AppStyles.apply_styles()
    
    # 한글 폰트 설정
    set_korean_font()
    
    # 탭 구성
    tabs = st.tabs(["🏠 소개", "🗺️ 여행지 추천", "👤 사용자 프로필"])
    
    # 여행 추천기 초기화
    recommender = TravelRecommender()
    
    # 사용자 입력 받기
    user_input = TravelerInputForm.get_input()
    
    # 페이지별 컨텐츠 표시
    with tabs[0]:  # 소개 페이지
        IntroPage.show()
    
    with tabs[1]:  # 여행지 추천 페이지
        st.markdown('<h1 class="main-header">🗺️ 여행지 추천</h1>', unsafe_allow_html=True)
        
        # 분석 시작 버튼 추가 (옵션)
        if st.button("🔍 내게 맞는 여행지 찾기", use_container_width=True):
            with st.spinner("여행 만족도 예측 중..."):
                # 추천 결과 계산
                results = recommender.recommend(user_input, top_n=20)
                
                # 결과 표시
                ResultVisualizer.display_results(results)
        else:
            # 버튼 클릭 전 안내 메시지
            st.info("좌측 사이드바에서 여행자 정보를 입력한 후, '내게 맞는 여행지 찾기' 버튼을 클릭하세요.")
            
            # 예제 사용자 프로필 표시
            st.markdown("""
            <div class="card">
                <h3>👋 여행 스타일 입력 가이드</h3>
                <p>
                    보다 정확한 추천을 위해 여행 스타일을 상세히 입력해 주세요. 
                    여행 스타일은 여행지 추천에 큰 영향을 미칩니다.
                </p>
                <ul>
                    <li><strong>자연 vs 도시</strong>: 자연 속에서 휴식을 즐기는 것을 선호하시나요, 아니면 도시의 활기찬 분위기를 즐기시나요?</li>
                    <li><strong>숙박 vs 당일</strong>: 현지에서 숙박하며 여유롭게 여행하는 것을 선호하시나요, 아니면 당일 여행을 선호하시나요?</li>
                    <li><strong>새로운지역 vs 익숙한지역</strong>: 새로운 지역을 탐험하는 것을 좋아하시나요, 아니면 익숙한 지역을 방문하는 것이 더 편하신가요?</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[2]:  # 사용자 프로필 페이지
        UserProfilePage.show(user_input)

if __name__ == "__main__":
    main() 