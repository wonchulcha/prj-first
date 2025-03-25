"""
여행 만족도 예측 시스템의 모델 관리 및 데이터 처리 모듈
"""

import pandas as pd
import numpy as np
import streamlit as st
from catboost import CatBoostRegressor, Pool
from typing import Dict, List, Tuple, Optional, Union
from constants import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, 
    DEFAULT_USER_INPUT, MODEL_PATH, DATA_PATH
)

class ModelManager:
    """
    CatBoost 모델을 관리하고 예측을 수행하는 클래스
    """
    def __init__(self, model_path: str = MODEL_PATH):
        """
        ModelManager 초기화
        
        Args:
            model_path: CatBoost 모델 파일 경로
        """
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> CatBoostRegressor:
        """
        CatBoost 모델 로드
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            로드된 CatBoost 모델
            
        Raises:
            Exception: 모델 로드 중 오류 발생 시
        """
        try:
            model = CatBoostRegressor()
            model.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"모델 로드 오류: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        주어진 데이터에 대한 만족도 예측 수행
        
        Args:
            data: 예측에 사용할 데이터프레임
            
        Returns:
            예측 만족도 배열
        """
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
                for col in missing_cols:
                    if col in NUMERICAL_FEATURES:
                        data_copy[col] = 0
                    else:
                        data_copy[col] = '0'
            
            # 필요한 컬럼만 선택하여 Pool 생성
            data_for_pool = data_copy[all_features].copy()
            
            # 데이터 타입 확인 및 변환
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
    """
    데이터 전처리를 담당하는 클래스
    """
    @staticmethod
    def prepare_input_data(input_dict: Dict, area: str) -> pd.DataFrame:
        """
        사용자 입력 데이터와 지역 정보를 결합하여 모델 입력용 데이터 생성
        
        Args:
            input_dict: 사용자 입력 정보
            area: 방문 지역명
            
        Returns:
            모델 입력용 데이터프레임
        """
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
            default_data = DEFAULT_USER_INPUT.copy()
            default_data['VISIT_AREA_NM'] = area
            return pd.DataFrame([default_data])
    
    @staticmethod
    def load_travel_data() -> Optional[pd.DataFrame]:
        """
        여행 데이터 로드
        
        Returns:
            여행 데이터 데이터프레임 또는 None (로드 실패 시)
        """
        try:
            return pd.read_csv(DATA_PATH)
        except Exception as e:
            st.error(f"데이터 로드 오류: {e}")
            return None


class TravelRecommender:
    """
    여행지 추천을 담당하는 클래스
    """
    def __init__(self):
        """TravelRecommender 초기화"""
        self.model_manager = ModelManager()
        self.data = DataPreprocessor.load_travel_data()
        
    def get_unique_areas(self, limit: int = 50) -> List[str]:
        """
        여행 데이터에서 고유한 지역 이름 추출
        
        Args:
            limit: 반환할 최대 지역 수
            
        Returns:
            지역 이름 리스트
        """
        if self.data is None:
            return []
        return self.data['VISIT_AREA_NM'].unique()[:limit].tolist()
    
    def recommend(self, user_input: Dict, top_n: int = 10) -> pd.DataFrame:
        """
        사용자 입력을 기반으로 여행지 추천
        
        Args:
            user_input: 사용자 입력 정보
            top_n: 반환할 추천 지역 수
            
        Returns:
            추천 결과 데이터프레임
        """
        if self.data is None:
            st.error("여행 데이터를 로드할 수 없습니다.")
            return pd.DataFrame()
        
        # 모든 지역에 대해 예측
        unique_areas = self.get_unique_areas()
        predictions = []
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, area in enumerate(unique_areas):
            # 상태 업데이트
            progress = (i + 1) / len(unique_areas)
            progress_bar.progress(progress)
            status_text.text(f"예측 진행 중... {i+1}/{len(unique_areas)} 지역")
            
            # 데이터 전처리
            input_df = DataPreprocessor.prepare_input_data(user_input, area)
            
            # 예측
            try:
                pred = self.model_manager.predict(input_df)[0]
                predictions.append({
                    'VISIT_AREA_NM': area,
                    '예측 만족도': round(pred, 2)
                })
            except Exception as e:
                st.error(f"{area} 지역 예측 오류: {e}")
                continue
        
        # 진행 상황 표시 제거
        progress_bar.empty()
        status_text.empty()
        
        # 결과가 없는 경우
        if not predictions:
            st.warning("예측 결과가 없습니다.")
            return pd.DataFrame()
        
        # 결과를 데이터프레임으로 변환하고 정렬
        results_df = pd.DataFrame(predictions)
        results_df = results_df.sort_values('예측 만족도', ascending=False)
        
        # 상위 N개 결과 반환
        return results_df.head(top_n) if top_n > 0 else results_df 