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

def carryover(x, strength, length=21):
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
    def __init__(self, country, target, metric=mape):
        """
            data: DataFrame containing both X and y
            target: (str) column in data that is the response variable
            metric: TBD
        """
        self.country = country
        self.target = target
        self.metric = metric
    
    def fit(self, X, y):
        """
            called immediately upon initialization of BayesianMixModel instance
            trains model
            X: channel media cost information
            y: response variable
        """
        
        self.X = X
        self.y = y
        
        with pm.Model() as mmm:
            channel_contributions = []
            
            data = pm.Data("data", self.X)
            
            for i, channel in enumerate(self.X.columns.values):
                coef = pm.Exponential(f'coef_{channel}', lam=0.0001)
                sat = pm.Exponential(f'sat_{channel}', lam=1)
                car = pm.Beta(f'car_{channel}', alpha=2, beta=2)

                channel_data = data.get_value()[:, i]
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

                channel_contributions.append(channel_contribution)

            base = pm.Exponential('base', lam=0.0001)
            noise = pm.Exponential('noise', lam=0.0001)

            sales = pm.Normal(
                'sales',
                mu=sum(channel_contributions) + base,
                sigma=noise,
                observed=y
            )

            trace = pm.sample(return_inferencedata=True, tune=3000)
        
        self.mmm = mmm
        self.trace = trace
        
    def predict(self, X):
        """
            X: DataFrame
        """
        pm.set_data({"data" : X}, model=self.mmm)
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
        scatterplot = pygal.XY(print_values=False, stroke=False, fill=False, height=500, width=1000, title="Model Predictions vs True Observations", x_title="actual", 
                                  y_title="predicted", explicit_size=True, show_legend=True, legend_at_bottom=True)
        
        x = self.y.values
        y = self.predict(self.X)

        scatterplot.add("data", [(x[i], y[i]) for i in range(len(x))])
        g = max(max(x), max(y))
        scatterplot.add("true = pred", [(0,0), (g, g)], stroke=True)
        show(scatterplot)