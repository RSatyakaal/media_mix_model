import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score 
from sklearn.metrics import mean_absolute_percentage_error

from datetime import datetime as dt
from functools import reduce
from yellowbrick.regressor import residuals_plot
from transformations import transform_one_column, select_best_feature, feature_imp_random_forest

import time
import ipywidgets as widgets
from ipywidgets import FileUpload
import datetime

from IPython.display import display
import io
import re
from scipy.optimize import minimize, LinearConstraint

import holidays

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score 
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from datetime import datetime as dt
from functools import reduce
from yellowbrick.regressor import residuals_plot
from transformations import transform_one_column, select_best_feature, feature_imp_random_forest

import time
import datetime
import ipywidgets as widgets
from ipywidgets import FileUpload

from IPython.display import display
import io
import re
from scipy.optimize import minimize, LinearConstraint

import holidays

import panel as pn
pn.extension()


def input_file(file_type, use_excel=False):
    """
        cell-1: read in the .csv file
    """
    
#     print(f"Welcome to MMM by SAAS Berkeley \nThis jupyter-notebook will take you through the steps necessary to produce a marketing mix model and a budget optimizer")
#     print(f"First, hit the UPLOAD button below, and attach the {file_type} (same format as before)")
#     print(f"This UPLOAD widget will only accept .csv or .xlsx formats, so please convert your {file_type} to .csv or .xslx BEFORE uploading")
    
    if not use_excel:
        upload = FileUpload(accept='.csv', multiple=False)
        display(upload)
    else:
        upload = FileUpload(accept='.xlsx', multiple=False)
        display(upload)
    
#     print(f"Once you upload the {file_type}, proceed by running the next cell")
    
    return upload
    

def parse_file(data, file_type, use_excel=False):
    """
        cell-2: parse the .csv file inputted into upload button
    """
    assert len(data.value) > 0, "PLEASE go back and run the above CELL"
    print(f"Thanks for uploading the {file_type}. I will display the dataframe in 2 seconds")
    uf = data.value[list(data.value.keys())[0]]['content']
    if not use_excel:
        df = pd.read_csv(io.BytesIO(uf))
    else:
        df = pd.read_excel(io.BytesIO(uf))
        
    display(df)
    t = input("Is the header correct [y/n]")
    
    if t == 'n' or t == 'N':
        z = int(input("Specify which row contains the header (zero-indexed)"))
        new_header = df.iloc[z]
        df = df.iloc[z+1:, :]
        df.columns = new_header
    
    
        display(df)    
        print(f"Type the letter y if the {file_type} file is good to go, OR the letter n if you'd like to reupload")
    
        x = input("proceed [y/n]")
        if x == 'y' or x == 'Y':
            print("SUCCESS -- PROCEED")
        elif x == 'n' or x == 'N':
            print("RERUN CELL-1")
    else:
        print("SUCCESS -- PROCEED")

    return df

def clean_sales(df, file_type, granularity=True):
    """
        cell-3: extract (1) country, (2) day, (3) response variable -- revenue, order_rate, etc.
    """
    
    print(f"COLUMNS of {file_type}: {df.columns.values}")
    
    
    
    def error_message():
        print("That column is not a valid column. We'll try again")
    
    def extract_country():
        print("If SALES DATA has more than one country that you'd like to segment and analyze separately, type the name of the column")
        print("If not, type NONE in that exact format")
        country = input("[column-name/NONE]")

        if "NONE" in country:
            country = []
        else:
            # 1. validate column
            if country not in df.columns.values:
                return extract_country()
        
        return [country]
    
    country = extract_country()
    
    # verify that no country values are missing
    assert df[country].isna().any().sum() == 0, f"The country column in this dataset contains null values. Please modify this and rerun the notebook."
    
    def extract_idx(n):
        print(f"Which column corresponds to the {n}")
        idx = input("[column-name]")
        
        if idx not in df.columns.values:
            error_message()
            return extract_idx(n)
        
        return [idx]
    
    
    if granularity:
        idx = extract_idx(n="day")
    else:
        idx = extract_idx(n="week")
    
    assert df[idx].isna().any().sum() == 0, f"The granularity/day/week column in this dataset contains null values. Please modify this and rerun the notebook."

    
    def extract_target():
        print(f"Which column corresponds to the target/response/dependent variable? In most cases, this will be REVENUE or something similar")
        
        y = input("[column-name]")
        
        if y not in df.columns.values:
            error_message()
            return extract_target()
        
        return [y]
    
    target = extract_target()
    
    assert df[target].isna().any().sum() == 0, f"The granularity/day/week column in this dataset contains null values. Please modify this and rerun the notebook."

    
    return df.set_index(idx)[country + target].sort_values(by=idx)

def create_pivot_table(table, num_col, indexes=['Publisher']):
        """
            cell-6 helper function: pivots input ad data to set index as day and features as media spending and impressions delivered
        """
        
        pivot_table =  pd.pivot_table(table, values=num_col, index=indexes,
                                      columns=['Day'], aggfunc=np.sum).T.fillna(0)
        if len(indexes) > 1:
            pivot_table.columns = list(map("_".join, pivot_table.columns))
        return pivot_table



def add_holidays(df, country):
    """
        inputs:
            df: dataframe with indexes as valid dates
            country: (str) country name 
        output:
            df: with added holiday indicator column
    """
    country = country.replace(" ", "")
    h = getattr(holidays, country)
    df["holiday"] = df.index.map(lambda x: int(x in h()))
    
    return df

def clean_and_merge(ad_df, sales_df):
    """
        cell-6: returns dictionary mapping country_name to ad+sales dataframe merged
        example: for proactiv, looks like {"US" : USA dataframe, "Canada" : Canada dataframe}
    """
    print(ad_df.columns.values)
    
    # LIMITATION: country column in sales data must be same as country column in ad data
    x = input("What column corresponds to the country column in the ad data?")
    y = input("What column corresponds to the media cost column in the ad data?")
    z = input("What column corresponds to the impressions column in the ad data?")
    keep = ad_df[x].notna()
    ad_df = ad_df[keep]
    countries = ad_df[x].unique().tolist()
    
    
    d = {}
    for country in countries:
        print(country)
        print(f"PROCESSING {country}")
        country_ad = ad_df[ad_df[x] == country]
        c_media_cost = create_pivot_table(country_ad, y, indexes=['Publisher']).add_suffix("_MediaCost")
        c_impressions = create_pivot_table(country_ad, z, ['Publisher']).add_suffix('_Impressions')
        c_ad = pd.concat([c_media_cost, c_impressions], axis=1,sort=False)
        c_sales = sales_df[sales_df[x] == country]
        
        if len(c_sales) == 0:
            print(f"The ad data and the sales data refer to {country} differently")
            print(f"How is {country} represented in the sales data?")
            
            cr = input(f"{country}")
            
            c_sales = sales_df[sales_df[x] == cr]
            if len(c_sales) == 0:
                print(f"That didn't work either. Please change the ad data or the sales data so that the values in the country/region column match.")
                return
        
#         print(len(c_ad), len(c_sales))
        final_c_df = c_ad.merge(c_sales, left_index=True, right_index=True, how='inner')
#         print("p0", len(final_c_df))

        final_c_df.drop(x, axis=1, inplace=True)
        
        if final_c_df.isna().values.any(): # fill in null values
            op = input("Your data has missing values. Do you want to 1. Delete rows with missing values 2. Reupload data in previous cell (and re-run all cells after) or 3. Fill null values with 0")
            if op == 1:
                final_c_df.dropna(inplace=True)
            elif op == 2:
                print("RERUN CELL-1")
                return 
            elif op == 3:
                final_c_df.fillna(0, inplace=True)
        
        if final_c_df.isna().any().sum() > 0:
            print("The dataset still contains null values. We are imputing zero for these values and proceeding. If this is an error, please modify the input dataset manually and rerun.")
        final_c_df.fillna(0, inplace=True)
        
#         print("p1", len(final_c_df))
        
        final_c_df = add_holidays(final_c_df, country) # add holidays
#         print("p2", len(final_c_df))

        final_c_df.fillna(0, inplace=True)
        d[country] = final_c_df
            
        print(f"MERGE SUCCESS for {country}")
    
    return d




### TRANSFORMATIONS



def carryover(x, alpha, L, theta=0, func='delayed'):
    '''
    inputs:
        x: vector of media spend
        alpha: in interval (0, 1) -- retention rate of ad effect of the m-th media from one period to the next aka decay parameter
        L: maximum duration of carryover effect assumed for a medium
        theta: in inteval (0, L) -- delay of the peak effect
        func: always use delayed adstock function
        
    output:
        returns transformed vector of spend accounting for carryover effect
    
    '''
    transformed_x = []
    if func=='geo':
        weights = geoDecay(alpha, L)
        
    elif func=='delayed':
        weights = delayed_adstock(alpha, theta, L)
        
    for t in range(x.shape[0]):
        upper_window = t+1
        lower_window = max(0,upper_window-L)
        current_window_x = x[:upper_window]
        t_in_window = len(current_window_x)
        if t < L:
            new_x = (current_window_x*np.flip(weights[:t_in_window], axis=0)).sum()
            transformed_x.append(new_x/weights[:t_in_window].sum())
        elif t >= L:
            current_window_x = x[upper_window-L:upper_window]
            ext_weights = np.flip(weights, axis=0) 
            new_x = (current_window_x*ext_weights).sum()
            transformed_x.append(new_x/ext_weights.sum())
            
    return np.array(transformed_x)

def delayed_adstock(alpha, theta, L):
    '''
    weighted average with delayed adstock function
    returns: weights of length L to calculate weighted averages with.
    
    alpha: in interval (0, 1) -- retention rate of ad effect from one period to the next aka decay parameter
    theta: in inteval (0, L) -- delay of the peak effect
    L: maximum duration of carryover effect assumed for a medium
    
    notes about L:
        1. it could vary for different media
        2. no prior information about L, it can be set to a very large number
    
    notes in general:
        1. adstock with geometric decay assumes advertising affect peaks at the same time as ad exposure
        2. delayed adstock function assumes some media may take longer to build up ad effect
    
    returns weights 
    
    '''
    return alpha**((np.ones(L).cumsum()-1)-theta)**2

def apply_adstock(df, column_name):
    """
        inputs:
            df: sales and ad data 
            column_name: (str) valid column name in string
            
        output:
            returns 1000 possible columns with unique transformations
    """
    
    df_transformations = pd.DataFrame()
    v = df[column_name].values
    
    for alpha in np.arange(0.1, 1, 0.1):
        for L in np.arange(0, 30, 2):
            for theta in [0]:
                col = f"{column_name}_alpha={alpha}L={L}theta={theta}"
                df_transformations[col] = carryover(x=v, alpha=alpha, L=L, theta=theta)
    
    df_transformations = df_transformations.set_index(df.index)
    
    return df_transformations

def best_adstock(df, response_var='Revenue'):
    """
        input:
            df: sales+ad dataframe
        output
            df_best: sales+ad dataframe with adstock transformation applied to each function
    """
    
    df_best = pd.DataFrame(index=df.index)
    media_vars = get_media_vars(df)
    
    for col in media_vars:
        tdf = apply_adstock(df, col) # df
        
#         tdf = transformed_df.merge(df[response_variable], left_index=True, right_index=True, how="inner")
        correlations = tdf.corrwith(df[response_var].astype(float))
    
        best_index = correlations.argmax()
    
        feature_name = correlations.index[best_index]
        r_value = correlations.iloc[best_index]
    
        print(f" {feature_name} || r-value of {r_value}")
        
        df_best[feature_name] = tdf[feature_name]
    
    
    
    return df_best

def hill_transform(x, ec, slope):
    # helper method for apply_diminishing_returns
    return 1 / (1 + (x / ec)**(-slope))

def apply_diminishing_returns(df, column_name, method='power'):
    """
        applies diminishing returns effect to every variable
        dr effect states that as media spend increases, the increase on incremental sales decreases
        
        input:
            df: data matrix
        output:
            transformed data matrix [power matrix]
    """
    
    df_transformations = pd.DataFrame()
    
    for power in np.arange(0.1, 0.7, 0.1):
        
                col = f"{column_name}_power={power}"
                df_transformations[col] = df[column_name] ** power
    
    df_transformations = df_transformations.set_index(df.index)
    
    return df_transformations

def best_diminishing_returns(df, response_var='Revenue'):
    df_best = pd.DataFrame(index=df.index)
    media_vars = get_media_vars(df)
    
    for col in media_vars:
        tdf = apply_diminishing_returns(df, col)
        
#         tdf = transformed_df.merge(df[response_variable], left_index=True, right_index=True, how="inner")
        correlations = tdf.corrwith(df[response_var].astype(float))
    
        best_index = correlations.argmax()
    
        feature_name = correlations.index[best_index]
        r_value = correlations.iloc[best_index]
    
        print(f" {feature_name} || r-value of {r_value}")
        
        df_best[feature_name] = tdf[feature_name]
    return df_best




def apply_transformations(data_dict, country, response_var='Revenue'):
    d = data_dict[country]
    d = d[get_media_vars(d) + get_impression_vars(d) + ['holiday', response_var]]
    
    # apply adstock/carryover
    tdf = best_adstock(d, response_var)
    imp_vars = get_impression_vars(d)
    df = pd.concat([tdf, d[[response_var]]], axis=1)
    # apply diminshing returns
    tdf = best_diminishing_returns(df, response_var)
    data_matrix = pd.concat([tdf, d[imp_vars + ['holiday', response_var]]], axis=1)
    
    print("number of nulls post transformations =", data_matrix.isna().any().sum())
    
    # fixes bug involving keyerror when column name has space in it
    data_matrix.columns = [col.replace(" ", "_") for col in data_matrix.columns]
    
    
    return data_matrix


def validate(model, X_train, Y_train, metricFunc):
    
    
    
    split = int(0.8 * len(X_train))
    
    
    xtrain, xval = X_train.iloc[:split, :], X_train.iloc[split:, :]
    ytrain, yval = Y_train.iloc[:split], Y_train.iloc[split:]

    model.fit(xtrain, ytrain)
    train_error = metricFunc(ytrain, model.predict(xtrain))
    val_error = metricFunc(yval, model.predict(xval))


    print(f"model={model} train_r2={train_error} validation_r2={val_error}")

    return val_error
    
class Scaler:
    def set_scaler(scaler_obj):
        Scaler.obj = scaler_obj
    
    def get_scaler():
        return Scaler.obj
    
def compute_CV(model, X_train, Y_train, metricFunc):
    '''
    Split the training data into  subsets.
    For each subset, 
        fit a model holding out that subset
        compute the MSE on that subset (the validation set)
    You should be fitting 4 models total.
    Return the average MSE of these 4 folds.

    Args:
        model: an sklearn model with fit and predict functions 
        X_train (data_frame): Training data
        Y_train (data_frame): Label 

    Return:
        the average validation MSE for the 4 splits.
    '''
    
    kf = KFold(n_splits=5)
    validation_errors = []
    
    for train_idx, valid_idx in kf.split(X_train):
        # split the data
        split_X_train, split_X_valid = X_train.iloc[train_idx,:], X_train.iloc[valid_idx,:]
        split_Y_train, split_Y_valid = Y_train.iloc[train_idx], Y_train.iloc[valid_idx]

        # Fit the model on the training split
        model.fit(split_X_train, split_Y_train)
        
        # Compute the RMSE on the validation split
        error = metricFunc(split_Y_valid, model.predict(split_X_valid))


        validation_errors.append(error)
        
    return np.mean(validation_errors)

def residual_plot(X, Y, model):
    """
        input:
            X: data matrix
            Y: response variable
            model: ridge regression model
    """
    plt.style.use('ggplot')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 24)
    viz = residuals_plot(model, X_train, y_train.to_numpy(), X_test, y_test.to_numpy())

    
def prep_mult_model(df, response_var='Revenue'):
    """
        input:
            df: multiplicative data matrix ready for modeling
    """
    X, Y = prep_additive_model(df, response_var)

    X = np.log(X + 1)
    Y = np.log(Y + 1)

    return X, Y


def prep_additive_model(df, response_var='Revenue'):
    """
        input:
            df: additive data matrix ready for modeling
        
    """
    X = df.drop([response_var], axis=1)
    Y = df[response_var]
    
    return X, Y





def ridge_regression(X, Y):
    """
        input:
            X: dataframe of features
            Y: dataframe/series of response variable (revenue)
        output:
            fitted ridge regression model
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 43)                            
    
#     scaler = StandardScaler()
#     X_train[X_train.columns] = scaler.fit_transform(X_train)
#     X_test[X_test.columns] = scaler.transform(X_test)
    
#     Scaler.set_scaler(scaler)
    
    
    
#     alphas = [0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
    alphas = [1/100000]
    scores =  []
    
    for a in alphas:
        model = Ridge(alpha=a, fit_intercept=True, positive=True)
        cv_score = validate(model, X_train, y_train, r2_score)
        scores.append(cv_score)
    i = np.array(scores).argmax()
    model = Ridge(alpha=alphas[i], fit_intercept=True, positive=True)

                           
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
#     X = pd.concat([X_train, X_test], axis=0)
    
    model.fit(X, Y)
    score = model.score(X, Y)
    residual_plot(X, Y, model)
    return model

def init_sliders(model):
    items =  []
#     items_layout = widgets.Layout(width='auto')
    col_to_slider = {}
    for col in model.feature_names_in_:
        
        if "MediaCost" in col:
            max_slider_val = 100000
        else:
            max_slider_val = 5000000

        style = {'description_width' : 'initial'}
        slider = widgets.IntSlider(0, 0, max_slider_val, 1, style=style)
        h = widgets.HBox([widgets.Label(col), slider])
        items.append(h)
        col_to_slider[col] = slider
        
    box_layout = widgets.Layout(display="flex", flex_flow="column", align_items="stretch", border="solid", width='80%')
        
    
    box = widgets.Box(children=items, layout=box_layout)
    return box, col_to_slider

def get_slider_vals(model, col_to_slider):
    return np.array([col_to_slider[col].value for col in model.feature_names_in_])

def mult_predict(origdf, model, col_to_slider):
    x = get_slider_vals(model, col_to_slider)
    
    shortened_names = [shorten_f_name(n) for n in model.feature_names_in_]
    print("The input is ", dict(zip(shortened_names, x)))
    
    carryover_vector = transform_row(x, model, origdf).to_numpy()
    v = []
    
    for i in range(len(model.feature_names_in_)):
        
        feature = model.feature_names_in_[i]
        
        if "MediaCost" in feature:
            
        
            power = float(extract_power(feature))
            orig_col, alpha, L, theta = re.findall("([\w|\s]+)_alpha=(\d.\d+)L=(\d+)theta=(\d+)", feature)[0]
            m = delayed_adstock(alpha=float(alpha), L=int(L), theta=int(theta))[0]
            temp = carryover_vector[i] + m * x[i]
            term = np.power(temp, power)
            v.append(term)
        else:
            v.append(x[i])

    v = np.array(v).reshape(1, -1)
    
    v = np.log(v + 1) # same thing as add_predict except with this last log transformation before we send it to the model
    
    y = model.predict(v)

    print(f"The prediction is {np.exp(y)}")

    return np.exp(y)

def add_predict(origdf, model, col_to_slider):
    x = get_slider_vals(model, col_to_slider)
    
    media_vars = [col for col in model.feature_names_in_ if "MediaCost" in col]
    
    
    carryover_vector = transform_row(x, model, origdf).to_numpy()
    v = []
    
    for i in range(len(model.feature_names_in_)):
        
        feature = model.feature_names_in_[i]
        
        if "MediaCost" in feature:
            
        
            power = float(extract_power(feature))
            orig_col, alpha, L, theta = re.findall("([\w|\s]+)_alpha=(\d.\d+)L=(\d+)theta=(\d+)", feature)[0]
            m = delayed_adstock(alpha=float(alpha), L=int(L), theta=int(theta))[0]
            temp = carryover_vector[i] + m * x[i]
            term = np.power(temp, power)
            v.append(term)
        else:
            v.append(x[i])
        

    v = np.array(v).reshape(1, -1)

    y = model.predict(v)

    print(f"The prediction is {y}")

    return y

def month_predict(forecast_df, origdf, model, model_type="additive"):
    """
        forecast_df: columns are shortened
        origdf: columns are
    """
    # COLUMNS MUST BE IN THE SAME ORDER AS REQUESTED FROM ABOVE
    
    
    forecast_df.columns = [shorten_f_name(col) + "_MediaCost" for col in forecast_df.columns] # fix the columns
    # print(forecast_df.columns)
    predictions = [] # list of predictions

    
    time = 0
    for time in range(len(forecast_df)):
        # define x to be today's set of payments
        x = forecast_df.iloc[time].to_numpy()
        carryover_vector = transform_row(x, model, origdf).to_numpy()
        v = []

        # build v up
        for i in range(len(model.feature_names_in_)):
        
            feature = model.feature_names_in_[i]

            if "MediaCost" in feature:


                power = float(extract_power(feature))
                orig_col, alpha, L, theta = re.findall("([\w|\s]+)_alpha=(\d.\d+)L=(\d+)theta=(\d+)", feature)[0]
                m = delayed_adstock(alpha=float(alpha), L=int(L), theta=int(theta))[0]
                temp = carryover_vector[i] + m * x[i]
                term = np.power(temp, power)
                v.append(term)
            else:
                v.append(x[i])
        # predict y from v
        v = np.array(v).reshape(1, -1)
        
        if model_type == "multiplicative":
            v = np.log(v + 1)
            y = np.exp(model.predict(v))
        else:
            y = model.predict(v)
            # add y to predictions
        predictions.append(y)
        #add row to origdf
        origdf.loc[origdf.index[-1] + datetime.timedelta(days=1)] = list(x) + list([0] * (origdf.shape[1] - len(x)))
        
        
    forecast_df['predictions'] = predictions
    return forecast_df

def show_saturation(model, col_to_slider):
    ans = []
    for fb in range(1, 100000, 10):
        arr = get_slider_vals(model, col_to_slider)
        
        arr[3] = 11661.14266113132 * hill_transform(fb, 11661.14266113132, 1)
        inp = np.log(arr + 1)
        
        prediction = model.predict(inp.reshape(1, -1))
        ans.append(np.exp(prediction))
    
    x = np.arange(0, 100000, 10)
    y = np.array(ans).flatten()
    sns.lineplot(x=x, y=y)
    plt.title()    
    
def get_media_vars(df):
    return [col for col in df.columns if "Media" in col and "Cost" in col]

def get_impression_vars(df):
    return [col for col in df.columns if "Impression" in col]

def transform_row(vector, model, origdf):
    """
        only does carryover effect
    """
    cols = origdf.columns.values
    
    row = {cols[i]: vector[i] for i in range(len(vector))}
    
    origdf = origdf.append(row, ignore_index=True)
    media_vars = [col for col in model.feature_names_in_ if "Media" in col and 'Cost' in col]
    
    for feature in media_vars:
        
        orig_col, alpha, L, theta = re.findall("([\w|\s]+)_alpha=(\d.\d+)L=(\d+)theta=(\d+)", feature)[0]
        origdf[feature] = carryover(x=origdf[orig_col].values, alpha=float(alpha), L=int(L), theta=int(theta))
        origdf.drop(orig_col, axis=1, inplace=True)
    
#     data_matrix = apply_diminishing_returns(origdf)[model.feature_names_in_]
    data_matrix = origdf[model.feature_names_in_]
    
    return data_matrix.iloc[-1, :]

def get_transformed_row(origdf, model, col_to_slider):
    """
        input:
            origdf: original dataframe pre transformations
            model: fitted model on data matrix post transformations
            col_to_slider: (dict) columns mapped to slider value
        output:
            vector, transformed (carryover + diminishing returns) data
    """
    idx = origdf.index[-1] + datetime.timedelta(days=1)
    
    new_index = [origdf.index] + [idx]
    
    last_row = get_slider_vals(model, col_to_slider) * 10000
    
    return transform_row(last_row, model, origdf)


def extract_power(colName):
    return re.findall("power=(\d.\d+)", colName)[0]


def add_sat_curve(model, i=3):
    col = model.feature_names_in_[i]
    coef = model.coef_[i]
    n = model.n_features_in_
    
    z = 100
    
    print(col, coef)
    
    xs = np.arange(0, 500000, 10)
    y = []
    
#     scaler = Scaler.get_scaler()
    
    for x in xs:
        t_x = z + x # adstock
        actual_x = t_x  ** float(extract_power(col)) # diminishing returns
        row = np.zeros(n)
        row[i] = actual_x
#         row = scaler.transform(row.reshape(1, -1))
        pred_y = model.predict(row.reshape(1, -1))
        y.append(pred_y)
        
    plt.style.use('fivethirtyeight')
    plt.plot(xs, y, color='blue');
    plt.title(f"Saturation Curve of {col[:10]}");
    plt.xlabel("Spending($)")
    plt.ylabel("Predicted Revenue ($)");
    

def mult_sat_curve(model, i=3):
    col = model.feature_names_in_[i]
    coef = model.coef_[i]
    n = model.n_features_in_
    
    z = 10000
    
    print(col, coef)
    
    xs = np.arange(0, 500000, 10)
    y = []
    
    for x in xs:
        t_x = z + x
        actual_x = t_x  ** float(extract_power(col))
        actual_x = np.log(actual_x)
        row = np.zeros(n)
        row[i] = actual_x
        pred_y = model.predict(row.reshape(1, -1))
        y.append(np.exp(pred_y))
    
    plt.style.use('fivethirtyeight')
    plt.plot(xs, y, color='blue');
    plt.title(f"Saturation Curve of {col}");
    plt.xlabel("Spending($)")
    plt.ylabel("Predicted Revenue ($)");
    
def pred_vs_true(model, X, Y, model_type="additive"):
    """
        inputs:
            model: fitted model
        
    """
    y_true = Y.values
    y_pred = model.predict(X)
    
    if model_type == "multiplicative":
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)
    
    mape = mean_absolute_percentage_error(y_true, y_pred) 
    plt.plot(X.index, y_true, color='blue', label='true revenue')
    plt.plot(X.index, y_pred, color='green', label='predicted revenue')
    plt.legend()
    plt.title(f"true vs predicted revenue (mape={mape}, model_type={model_type})");
    
def pred_vs_true_v2(model, X, Y, model_type="additive"):
    
    split = int(0.8 * len(X_train))
    
    
    xtrain, xval = X_train.iloc[:split, :], X_train.iloc[split:, :]
    ytrain, yval = Y_train.iloc[:split], Y_train.iloc[split:]

    model.fit(xtrain, ytrain)
    train_error = metricFunc(ytrain, model.predict(xtrain))
    val_error = metricFunc(yval, model.predict(xval))


    print(f"model={model} train_r2={train_error} validation_r2={val_error}")
    

def set_bounds(model):
    items =  []
#     items_layout = widgets.Layout(width='auto')
    col_to_slider = {}
    h = pn.Column()
    for col in model.feature_names_in_:
        
        if "MediaCost" in col:
            style = {'description_width' : 'initial'}
            slider = pn.widgets.IntRangeSlider(name=f"{col[:15]}_bounds", start=0, end=50000, step=1)
            s = pn.Row("", slider)
            h.extend(s)
            col_to_slider[col] = slider
            
    return h, col_to_slider

def get_bound_vals(model, col_to_slider):
    
    media_vars = [fn for fn in model.feature_names_in_ if 'MediaCost' in fn]
    
    return np.array([col_to_slider[col].value for col in media_vars]) + 1

# def mult_optimize_budget(origdf, data_matrix, model, col_to_slider, budget=20000):
#     num_media_vars = len(get_media_vars(data_matrix))
#     var_names = model.feature_names_in_[:num_media_vars]
#     num_vars = num_media_vars
#     coef = model.coef_[:num_media_vars]
#     vector = np.zeros(18)
#     carryover_vector = transform_row(vector, model, origdf).to_numpy()[:num_media_vars]
#     def objective_function(x, weights):
#         return -1 * np.log((carryover_vector + x)**0.05 + np.ones(num_vars)).dot(weights)
    
#     return optimize_budget(origdf, data_matrix, model, objective_function, col_to_slider, budget)




def shorten(lst):
    return [word[:10] for word in lst]

def shorten_f_name(string):
    end = 0
    while end < len(string) and string[end] != "_":
        end += 1
    
    return string[:end]

def graph(dct, title):
    df = pd.DataFrame(data= {'variable' : shorten(list(dct.keys())), 'spend' : list(dct.values())})
    sns.barplot(data=df, x = 'variable', y = 'spend')
    plt.xticks(rotation = 45);
    plt.title(title);
    
def add_optimize_month(origdf, data_matrix, model, col_to_slider, budget=20000):
    list_of_budgets = []    
    # append budget1 to origdf
    for day in np.arange(1, 31, 1):
        budget_mix = add_optimize_budget(origdf, data_matrix, model, col_to_slider, budget=budget)
#         print("budgetmix", budget_mix.values())
        budget_values = list(budget_mix.values()) + [0] * (origdf.shape[1] - len(budget_mix.values()))
        list_of_budgets.append(budget_values)

#         dictlike = dict(zip(get_media_vars(origdf), budget_values))
#         origdf = origdf.append()
#         print(origdf.iloc[-1, :].values)
        before = len(origdf)
        origdf.loc[origdf.index[-1] + datetime.timedelta(days=1)] = budget_values
        after = len(origdf)
        
    
    return list_of_budgets

def add_optimize_budget(origdf, data_matrix, model, col_to_slider, budget=20000):
    num_media_vars = len(get_media_vars(data_matrix))
    var_names = model.feature_names_in_[:num_media_vars]
    vector = np.zeros(origdf.shape[1])
    carryover_vector = transform_row(vector, model, origdf).to_numpy()[:num_media_vars]
    return optimize_budget(origdf, data_matrix, model, col_to_slider, budget)


def optimize_budget(origdf, data_matrix, model, col_to_slider, budget=20000):
    num_media_vars = len(get_media_vars(data_matrix))
    var_names = model.feature_names_in_[:num_media_vars]
    num_vars = num_media_vars
    coef = model.coef_[:num_media_vars]
    vector = np.zeros(origdf.shape[1])
    carryover_vector = transform_row(vector, model, origdf).to_numpy()[:num_media_vars]
    constraint = LinearConstraint(np.ones(num_vars), lb=budget, ub=budget)
    
    bounds_from_slider = get_bound_vals(model, col_to_slider)
    
    bounds = [(bounds_from_slider[i][0], min(budget, bounds_from_slider[i][1])) for i in range(num_vars)]
    
    def objective_function(x, weights):
        
        powers = [float(extract_power(colName)) for colName in model.feature_names_in_]
        
        v = np.array([])
        for i, p in enumerate(powers):
            feature = model.feature_names_in_[i]
            orig_col, alpha, L, theta = re.findall("([\w|\s]+)_alpha=(\d.\d+)L=(\d+)theta=(\d+)", feature)[0]
            m = delayed_adstock(alpha=float(alpha), L=int(L), theta=int(theta))[0]
            temp = carryover_vector[i] + m * x[i]
            term = np.power(temp, p)

            v = np.append(v, term)
        return -1 * v.dot(weights)
    
    res = minimize(
    objective_function, method="SLSQP",
    x0 = budget * np.random.random(num_vars),
    args=(coef),
    constraints=constraint,
    bounds=bounds)
    
    print(f"Budget = ${budget}, Expected Revenue/Orders = {-1 * res.fun}")
    optimized_budget = dict(zip(var_names, res.x))
    return optimized_budget

def graph_month(llb, x_opt):
    llb_clipped = [el[:x_opt.shape[1]] for el in llb]
    llb_df = pd.DataFrame(data=llb_clipped, columns=x_opt.columns)

    plt.style.use('ggplot')
    sns.lineplot(data=llb_df)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");
    plt.title("Optimized Budget for Month")
    plt.xlabel("Days")
    plt.ylabel("Spending ($)");
    

def export_attribute_table(model):
    coef = model.coef_
    
    
    
    
