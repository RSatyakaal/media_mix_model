import pandas as pd
import numpy as np

import pymc3 as pm
import arviz as az
import theano.tensor as tt

import matplotlib.pyplot as plt
import seaborn as sns
import pygal
from IPython.display import SVG, display

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split

from helper import *


def saturate(x, a):
    """
        arbitrary saturation curve, parameters of this function must define saturation curve
    """
    return 1 - tt.exp(-a*x)

def carryover(x, strength, length=10):
    """
        same function as specified in google whitepaper
        usually use poission random variable for length
    """
    w = tt.as_tensor_variable(
        [tt.power(strength, i) for i in range(length)]
    )
    
    x_lags = tt.stack(
        [tt.concatenate([
            tt.zeros(i),
            x[:x.shape[0]-i]
        ]) for i in range(length)]
    )
    
    return tt.dot(w, x_lags)

def show(chart):
    display(SVG(chart.render(disable_xml_declaration=True)))

class BayesianMixModel:
    def __init__(self, client, country, target, metric=mape):
        """
            data: DataFrame containing both X and y
            target: (str) column in data that is the response variable
            metric: TBD
        """
        self.client = client
        self.country = country
        self.target = target
        self.metric = metric
    
    def fit(self, X, y, tune=3000):
        """
            called immediately upon initialization of BayesianMixModel instance
            trains model
            X: channel media cost information
            y: response variable
        """
        
        self.X = X
        self.y = y
        # implementing the sklearn Estimator interface
        self.feature_names_in_ = X.columns.values
        self.n_features_in_ = len(self.feature_names_in_)
        self.n_iter = None
        
        
        
        with pm.Model() as mmm:
            channel_contributions = []
            
#             data = pm.Data("data", self.X)
            
            for i, channel in enumerate(self.X.columns.values):
                coef = pm.Exponential(f'coef_{channel}', lam=0.0001)
                sat = pm.Exponential(f'sat_{channel}', lam=1)
                car = pm.Beta(f'car_{channel}', alpha=2, beta=2)

#                 channel_data = data.get_value()[:, i]
#                 print(channel)
                channel_data = pm.Data(channel, self.X.iloc[:, i].values)
                channel_contribution = pm.Deterministic(
                    f'contribution_{channel}',
                    coef * saturate(
                        carryover(
                            channel_data,
                            car
                        ),
                        sat
                    )
                )
                # uncomment out the entire above line

                channel_contributions.append(channel_contribution)
                # change above line when done testing to .append(channel_contribution)

            base = pm.Exponential('base', lam=0.0001)
            noise = pm.Exponential('noise', lam=0.0001)
            coef = pm.Normal("coef", mu=0, sigma=10)
            sales = pm.Normal(
                'sales',
                mu = sum(channel_contributions) + base,
                sigma=noise,
                observed=y
            )

            trace = pm.sample(return_inferencedata=True, tune=tune)
        
        self.mmm = mmm
        self.trace = trace
        
    def predict(self, X):
        """
            X: DataFrame
        """
        with self.mmm:
            pm.set_data({channel: X.iloc[:, i].values for i, channel in enumerate(X.columns.values)}, model=self.mmm)
            ppc_test = pm.sample_posterior_predictive(self.trace, model=self.mmm, samples=1000)
            p_test_pred = ppc_test["sales"].mean(axis=0)
        
        return p_test_pred
    
    def score(self, X, y):
        """
            X: DataFrame
            y: Series
        """
        if self.metric:
            return metric(self.predict(X), y)
        else:
            return mape(self.predict(X), y)

    
    def lineplot(self):
        """
            plots actual vs fitted time series on entire training set
        """
        means = self.predict(self.X)

        line_chart = pygal.Line(fill=False, height=500, width=1000, title="Model Fit Time Series", x_title="Day", 
                              y_title=f"{self.target}", explicit_size=True, show_legend=True, legend_at_bottom=False)
        line_chart.add('TRUE', self.y.values)
        line_chart.add("PREDICTION", means)
        show(line_chart)

    
    def scatterplot(self):
        """
            plots actual vs fitted time series
        """
        scatterplot = pygal.XY(print_values=False, stroke=False, fill=False, height=500, width=1000, title="[Training] Model Predictions vs True Observations", x_title="actual", 
                                  y_title="predicted", explicit_size=True, show_legend=True, legend_at_bottom=True)
        
        x = self.y.values
        y = self.predict(self.X)

        scatterplot.add("data", [(x[i], y[i]) for i in range(len(x))])
        g = max(max(x), max(y))
        scatterplot.add("true = pred", [(0,0), (g, g)], stroke=True)
        show(scatterplot)
    
        
    def attribution(self):
        """
            inputs:
                target - (str) response variable
            output:
                attribution graph
        """
        def compute_mean(trace, channel):
                        
            return (trace
                    .posterior[f'contribution_{channel}']
                    .values
                    .reshape(4000, len(self.X))
                    .mean(0)
                   )
        target = self.target
        X = self.X
        y = self.y
        trace = self.trace
        data = self.X
        
        channels = X.columns.values
        unadj_contributions = pd.DataFrame(
            {'Base': trace.posterior['base'].values.mean()},
            index=X.index
        )
        for channel in channels:
            unadj_contributions[channel] = compute_mean(trace, channel)
        adj_contributions = (unadj_contributions
                             .div(unadj_contributions.sum(axis=1), axis=0)
                             .mul(y, axis=0)
                            )
        attribution_table = pd.DataFrame({'Revenue Contributions': adj_contributions.sum(axis=0)[1:], 
                                          'Media Spending': data[channels].sum(axis=0)})
        attribution_table['Attribution'] = attribution_table['Revenue Contributions'] / attribution_table['Media Spending']
        attribution_table.to_excel(f'{self.client}_attribution_table.xlsx')

        ax = (adj_contributions
          .plot.area(
              figsize=(16, 10),
              linewidth=1,
              title=f'Predicted {target} and Breakdown',
              ylabel=target,
              xlabel='Date'
          )
         )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1], labels[::-1],
            title='Channels', loc="center left",
            bbox_to_anchor=(1.01, 0.5)
        )

        '''line_chart = pygal.StackedLine(fill=True, explicit_size=True, height=600, width=1000, legend_at_bottom=True, title="Attribution", x_title="Day", y_title=f"{target}")
        for col in adj_contributions.columns:
            line_chart.add(col, adj_contributions[col].values)
        show(line_chart)'''
        
    def saturation_curve(self):
        """
            outputs len(saturation_columns) saturation curves
        """
        plt.style.use('ggplot')
        x = np.arange(0, 100, 1)
        saturation_columns = self.X.columns.values
        trace = self.trace

        fig, axs = plt.subplots(len(saturation_columns))
        for i in range(0, len(saturation_columns)):
            sat_values = trace.posterior[f"sat_{saturation_columns[i]}"].values.flatten()
            q1 = np.percentile(sat_values, q=2.5)
            q2 = np.percentile(sat_values, q=97.5)
            a = sat_values.mean()

            axs[i].plot(x,1 - np.exp(-a * x))
            axs[i].fill_between(x, 1 - np.exp(-q1 * x), 1 - np.exp(-q2 * x), color='b', alpha=.1)
            axs[i].set_title(f"Saturation Curve for {shorten_f_name(saturation_columns[i]).upper()}")
        fig.suptitle(f"Saturation Curves");
        for ax in axs.flat:
            ax.set(xlabel=f"carryover ($)", ylabel = "saturation proportion")
        
# helper methods

def calculate_mape(model, xtrain, ytrain, xval, yval):
    """
        inputs are self-explanatory, simply predicts mean absolute percentage error on training set and validation set
    """
    trainPred = model.predict(xtrain)
    valPred = model.predict(xval)
    val_mape = mape(yval, valPred)
    train_mape = mape(ytrain, trainPred)
    
    return train_mape, val_mape
        
def validation_scatterplot(model, X, y):
    """
        plots actual vs fitted time series
    """
    scatterplot = pygal.XY(print_values=False, stroke=False, fill=False, height=500, width=1000, title="[Validation] Model Predictions vs True Observations", x_title="actual", 
                              y_title="predicted", explicit_size=True, show_legend=True, legend_at_bottom=True)

    x = y.values
    y = model.predict(X)

    scatterplot.add("data", [(x[i], y[i]) for i in range(len(x))])
    g = max(max(x), max(y))
    scatterplot.add("true = pred", [(0,0), (g, g)], stroke=True)
    show(scatterplot)
    
def validation_lineplot(model, X, y, target):
    """
        plots actual vs fitted time series on entire training set
    """
    means = model.predict(X)

    line_chart = pygal.Line(fill=False, height=500, width=1000, title="[Validation] Model Fit Time Series", x_title="Day", 
                          y_title=f"{target}", explicit_size=True, show_legend=True, legend_at_bottom=False)
    line_chart.add('TRUE', y.values)
    line_chart.add("PREDICTION", means)
    show(line_chart)
    

        
        
        
# OPTIMIZATION HELPER METHODS
def gen_samples(model):
    """
        returns a single sample of the posterior
        for use in optimization
    """
    coef = []
    sat = []
    car = []
    
    sample_num = np.random.randint(low=0, high=999)
    
    for channel in model.feature_names_in_:
        
        posterior = model.trace.posterior
        coef.append(posterior[f"coef_{channel}"].to_numpy().mean(axis=0).flatten()[sample_num])
        sat.append(posterior[f"sat_{channel}"].to_numpy().mean(axis=0).flatten()[sample_num])
        car.append(posterior[f"car_{channel}"].to_numpy().mean(axis=0).flatten()[sample_num])
        
    return np.array(coef), np.array(sat), np.array(car)

def get_means(model):
    """
        returns the means of the posterior
        for use in mean optimization
    """
    coef = []
    sat = []
    car = []
    
    sample_num = np.random.randint(low=0, high=999)
    
    for channel in model.feature_names_in_:
        
        posterior = model.trace.posterior
        coef.append(posterior[f"coef_{channel}"].to_numpy().mean())
        sat.append(posterior[f"sat_{channel}"].to_numpy().mean())
        car.append(posterior[f"car_{channel}"].to_numpy().mean())
        
    return np.array(coef), np.array(sat), np.array(car)

def create_dataset_extra_rows(model):
    """
        returns a properly pre-processed dataset
        attaches past 10 rows from training dataset
        adds a dummy row at the bottom because last carryover doesn't matter
    """
    last_10_rows = model.X.tail(10)   
    last_10_rows.loc[last_10_rows.index[-1] + datetime.timedelta(days=1)] = 0
    return last_10_rows

def calculate_carryover_vector(model, car, last_10_rows):
    """
        calculates carryover vector by iterating over each channel, and running carryover on each channel given the sampled car value
    """
    cv = []
    post = model.trace.posterior
    for idx, channel in enumerate(model.feature_names_in_):
        column = last_10_rows[channel].values
        param = car[idx]
        value = carryover(column, param).eval().flatten()[-1]
        cv.append(value)
    
    return np.array(cv)

def objective_function(x, coef, sat, car, carryover_vector):
    """
        this is after values are sampled from the posterior
        x is an array of raw spends
        coef, sat, car, carryover should be vectors of length c where c = number of channels
        carryover should contain the carryover from past (10) days
        
        all should be numpy arrays
    """
    c = len(coef)
    # create matrix of coefficients for final step
    weights = np.diag(coef)
    
    # assuming we pre-compute carryovers so we can exclude from objective fn
    # note: there is no loss at day-0 of spends
    spends = carryover_vector + x
    
    # applies saturation curve on carryover'd spend
    sats = np.diag(sat)
    temp = 1 - np.exp(-sats@spends)
    # since our optimizer is a minimizer, we need to negate our predicted #orders
    y = weights@temp
    

    return -1*np.sum(weights@temp)

def optimize(model, n_iter=1000, budget=20000):
    """
        model: BayesianMixModel already trained
        n_iter: (int) default=1000
        budget: (int) default=20000
    """
    xs = []
    c = model.n_features_in_
    for i in range(n_iter):
        
        # (1) sample parameter values
        coef, sat, car = gen_samples(model)
#         print("PARAM:", coef, sat, car)
        last_10_rows = create_dataset_extra_rows(model)
        
        # (2) calculate the carryover vector
        cv = calculate_carryover_vector(model, car, last_10_rows)
#         cv = np.zeros(c)
        # (3) run the optimizer [same as before, just modified to work]
        # note: change the constraints and bounds later
        constraint = LinearConstraint(np.ones(c), lb=0, ub=budget)
        res = minimize(objective_function, method="trust-constr", x0 = budget/c * np.random.random(c), args=(coef, sat, car, cv), constraints=constraint, bounds=[(0, 2*budget/c) for j in range(c)])
        xs.append(res.x)
    
    return np.array(xs)

def optimize_using_mean(model, budget):
    xs = []
    c = model.n_features_in_
        
    # (1) sample parameter values
    coef, sat, car = get_means(model)
#         print("PARAM:", coef, sat, car)
    last_10_rows = create_dataset_extra_rows(model)

    # (2) calculate the carryover vector
    cv = calculate_carryover_vector(model, car, last_10_rows)
    # (3) run the optimizer [same as before, just modified to work]
    # note: change the constraints and bounds later
    constraint = LinearConstraint(np.ones(c), lb=0, ub=budget)
    res = minimize(objective_function, method="trust-constr", x0 = budget/c * np.random.random(c), args=(coef, sat, car, cv), constraints=constraint, bounds=[(0, 2*budget/c) for j in range(c)])
    xs.append(res.x)
    
    return np.array(xs)


def bayesian_optimize_month(model, budget, days=30):
    """
        optimizes for 30 days
    """
    last_10_rows = create_dataset_extra_rows(model)
    xs = []
    c = model.n_features_in_
    for day in range(days):
        # (1) sample parameter values
        coef, sat, car = get_means(model)
        # (2) calculate the carryover vector
        cv = calculate_carryover_vector(model, car, last_10_rows)
        # (3) run the optimizer [same as before, just modified to work]
        # note: change the constraints and bounds later
        constraint = LinearConstraint(np.ones(c), lb=0, ub=budget)
        res = minimize(objective_function, method="trust-constr", x0 = budget/c * np.random.random(c), args=(coef, sat, car, cv), constraints=constraint, bounds=[(0, budget/2) for j in range(c)])
        xs.append(res.x)
        last_10_rows.loc[last_10_rows.index[-1] + datetime.timedelta(days=1)] = res.x    
    
    
    # write optimized month results to file
    df = pd.DataFrame(columns=initial_model.feature_names_in_, data=np.array(xs))
    df.to_excel("optimized_month.xlsx", header=True, index=True)
    
    return np.array(xs)
    