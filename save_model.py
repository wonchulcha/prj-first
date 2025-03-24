import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def main():
    
    df_place = load_data('./data/tn_visit_area_info.csv')
    df_travel = load_data('./data/tn_travel.csv')
    df_traveler = load_data('./data/tn_traveller_master.csv')

    df = pd.merge(df_place, df_travel, on='TRAVEL_ID', how='left')
    df = pd.merge(df, df_traveler, on='TRAVELER_ID', how='left')

    df_filter = df[~df['TRAVEL_MISSION_CHECK'].isnull()].copy()
    df_filter.loc[:, 'TRAVEL_MISSION_INT'] = df_filter['TRAVEL_MISSION_CHECK'].str.split(';').str[0].astype(int)

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

    df_filter = df_filter.dropna()

    categorical_features_names = [
        'GENDER',
        # 'AGE_GRP',
        'TRAVEL_STYL_1',
        'TRAVEL_STYL_2',
        'TRAVEL_STYL_3',
        'TRAVEL_STYL_4',
        'TRAVEL_STYL_5',
        'TRAVEL_STYL_6',
        'TRAVEL_STYL_7',
        'TRAVEL_STYL_8',
        'TRAVEL_MOTIVE_1',
        # 'TRAVEL_COMPANIONS_NUM',
        'TRAVEL_MISSION_INT',
        'VISIT_AREA_NM',
        # 'DGSTFN'
    ]


    df_filter[categorical_features_names[1:-1]] = df_filter[categorical_features_names[1:-1]].astype(int)

    train_df, test_df = train_test_split(df_filter, test_size=0.2, random_state=42)

    train_pool = Pool(
        train_df.drop('DGSTFN', axis=1),
        train_df['DGSTFN'],
        cat_features=categorical_features_names
    )

    test_pool = Pool(
        test_df.drop('DGSTFN', axis=1),
        test_df['DGSTFN'],
        cat_features=categorical_features_names
    )

    model = CatBoostRegressor(
        loss_function='RMSE',
        eval_metric='MAE',
        task_type='CPU',
        depth=6,
        learning_rate=0.01,
        n_estimators=2000
    )

    model.fit(
        train_pool,
        eval_set=test_pool,
        verbose=500,
        plot=True
    )

    df_filter.to_csv('./data/df_filter.csv', encoding='utf-8', index=False)
    model.save_model('catboost_model.cbm')

if __name__ == '__main__':
    main()
