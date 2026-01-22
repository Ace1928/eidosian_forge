from pyomo.common.dependencies import pandas as pd
import pyomo.environ as pyo
This function generates an instance of the rooney & biegler Pyomo model

    Returns
    -------
    m: an instance of the Pyomo model
        for uncertainty propagation
    