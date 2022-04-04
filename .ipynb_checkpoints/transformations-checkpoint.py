import pandas as pd
import numpy as np

def transform_one_column(df, column_name, alpha_values=[0.6, 0.7, 0.8, 0.9, 1], decay_values=[0.6, 0.7, 0.8, 0.9, 1], lag=5):
    """
        inputs:
            df: dataframe containing channel mix information
                example: /data/proactiv/clean/'US Publisher Media Cost with Sales.csv'
            alpha_values: list-like containing possible alpha values
                alpha is a hyperparameter which describes ad saturation levels and the idea of diminishing returns
            decay_values: list-like containing possible decay values
                the “retention rate” of which advertising has an impact on periods after the ad took place
            lag: (int) estimated difference between max ad impact and week of ad exposure
            
        outputs:
            df_col_lag_alpha_decay: dataframe containing channel mix information with respective three transformations applied
    """
    # APPLY LAG TRANSFORMATION BY SHIFTING COLUMNS DOWN
    df_col = df[column_name]
    df_col_lag = pd.concat([df_col] * lag, axis = 1, ignore_index=True).rename(lambda x: column_name + 'Lag' + str(x), axis=1)
    for i in np.arange(1, lag):
        new_column_name = column_name + f"Lag{i}"
        df_col_lag[new_column_name] = df_col_lag[new_column_name].shift(i, fill_value=0)
        

    l = []
    for alpha in alpha_values:
        
        c = df_col_lag.copy()
        for column in c.columns:
            new_column_name = column + f"Alpha{alpha}"
            c[new_column_name] = c[column]**alpha
            c.drop(columns=[column], inplace=True)
        l.append(c)
    
    df_col_lag_alpha = pd.concat(l, axis=1, ignore_index=False)
    
    # APPLY DECAY TRANSFORMATION
    l = []
    for decay in decay_values:
        
        c = df_col_lag_alpha.copy()
        for column in c.columns:
            new_column_name = column + f"Decay{decay}"
            c[new_column_name] = c[column] # initialize a new series column
            
            rows = len(c[new_column_name])
            for i in np.arange(0, rows): # iterate over every value, performing the decay transformation
                if i == 0:
                    last_week_activity = 0
                else:
                    last_week_activity = c[new_column_name].iloc[i-1]
                
                this_week_activity = c[new_column_name].iloc[i]
                c[new_column_name].iloc[i] = decay*this_week_activity + (1 - decay)*last_week_activity
            
            c.drop(columns=[column], inplace=True)
        l.append(c)
    
    df_col_lag_alpha_decay = pd.concat(l, axis=1, ignore_index=False)
    
    return df_col_lag_alpha_decay
            
                

        
def select_best_feature(df, transformed_df, response_variable = "Revenue"):
    """
        inputs:
            transformed_df: dataframe with lag, alpha, decay transformations applied for the column
        
        output:
            list-like of column_names in transformed_df with the highest correlation to the sales variable
    """
    
    tdf = transformed_df.merge(df[response_variable], left_index=True, right_index=True, how="inner")
    correlations = tdf.corr()[response_variable]
    
    
#     best_index = None
#     best_val = float('inf')
#     for idx, val in enumerate(correlations.drop("Revenue")):
#         if abs(val) > best_val:
#             best_val = val
#             best_index = idx
    
    
    best_index = np.abs(correlations.drop("Revenue")).argmax()
    
    feature_name = correlations.index[best_index]
    r_value = correlations.iloc[best_index]
    
    print(f"The most indicative media variable is {feature_name} with an r-value of {r_value}")
    
    
    return transformed_df[feature_name]


# Function using Random Forest Feature Importances to Output a Series of the Most Important Feature for a Given Column

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def feature_imp_random_forest(df, transformed_df, response_variable = "Revenue", absol = True):
    transformed_df = transformed_df.merge(df["Revenue"], left_index=True, right_index=True, how="inner")
    X = transformed_df.drop(['Revenue'], axis=1)
    Y = transformed_df[['Revenue']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    columns = X_train.columns
    coefficients = model.feature_importances_.reshape(X_train.columns.shape[0], 1)
    if absol:
        coefficients = abs(coefficients)
    df = pd.concat((pd.DataFrame(columns, columns = ['Feature']), pd.DataFrame(coefficients, columns = ['Coefficient'])), axis = 1).sort_values(by='Coefficient', ascending = False)
    best_feature = df['Feature'].iloc[0]
    return transformed_df[best_feature]