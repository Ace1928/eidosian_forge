from pyomo.common.dependencies import numpy as np, pandas as pd
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
def reactor_design_model_for_datarec(data):
    model = reactor_design_model(data)
    model.caf.fixed = False
    return model