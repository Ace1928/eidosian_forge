from pyomo.common.dependencies import pandas as pd
import pyomo.environ as pyo
def rooney_biegler_model(data):
    """This function generates an instance of the rooney & biegler Pyomo model using 'data' as the input argument

    Parameters
    ----------
    data: pandas DataFrame, list of dictionaries, or list of json file names
        Data that is used to build an instance of the Pyomo model

    Returns
    -------
    m: an instance of the Pyomo model
        for estimating parameters and covariance
    """
    model = pyo.ConcreteModel()
    model.asymptote = pyo.Var(initialize=15)
    model.rate_constant = pyo.Var(initialize=0.5)

    def response_rule(m, h):
        expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
        return expr
    model.response_function = pyo.Expression(data.hour, rule=response_rule)

    def SSE_rule(m):
        return sum(((data.y[i] - m.response_function[data.hour[i]]) ** 2 for i in data.index))
    model.SSE = pyo.Objective(rule=SSE_rule, sense=pyo.minimize)
    return model