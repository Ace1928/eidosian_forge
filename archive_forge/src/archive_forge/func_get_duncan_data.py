import json
import os
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
def get_duncan_data():
    labels = ['accountant', 'pilot', 'architect', 'author', 'chemist', 'minister', 'professor', 'dentist', 'reporter', 'engineer', 'undertaker', 'lawyer', 'physician', 'welfare.worker', 'teacher', 'conductor', 'contractor', 'factory.owner', 'store.manager', 'banker', 'bookkeeper', 'mail.carrier', 'insurance.agent', 'store.clerk', 'carpenter', 'electrician', 'RR.engineer', 'machinist', 'auto.repairman', 'plumber', 'gas.stn.attendant', 'coal.miner', 'streetcar.motorman', 'taxi.driver', 'truck.driver', 'machine.operator', 'barber', 'bartender', 'shoe.shiner', 'cook', 'soda.clerk', 'watchman', 'janitor', 'policeman', 'waiter']
    exog = [[1.0, 62.0, 86.0], [1.0, 72.0, 76.0], [1.0, 75.0, 92.0], [1.0, 55.0, 90.0], [1.0, 64.0, 86.0], [1.0, 21.0, 84.0], [1.0, 64.0, 93.0], [1.0, 80.0, 100.0], [1.0, 67.0, 87.0], [1.0, 72.0, 86.0], [1.0, 42.0, 74.0], [1.0, 76.0, 98.0], [1.0, 76.0, 97.0], [1.0, 41.0, 84.0], [1.0, 48.0, 91.0], [1.0, 76.0, 34.0], [1.0, 53.0, 45.0], [1.0, 60.0, 56.0], [1.0, 42.0, 44.0], [1.0, 78.0, 82.0], [1.0, 29.0, 72.0], [1.0, 48.0, 55.0], [1.0, 55.0, 71.0], [1.0, 29.0, 50.0], [1.0, 21.0, 23.0], [1.0, 47.0, 39.0], [1.0, 81.0, 28.0], [1.0, 36.0, 32.0], [1.0, 22.0, 22.0], [1.0, 44.0, 25.0], [1.0, 15.0, 29.0], [1.0, 7.0, 7.0], [1.0, 42.0, 26.0], [1.0, 9.0, 19.0], [1.0, 21.0, 15.0], [1.0, 21.0, 20.0], [1.0, 16.0, 26.0], [1.0, 16.0, 28.0], [1.0, 9.0, 17.0], [1.0, 14.0, 22.0], [1.0, 12.0, 30.0], [1.0, 17.0, 25.0], [1.0, 7.0, 20.0], [1.0, 34.0, 47.0], [1.0, 8.0, 32.0]]
    endog = [82.0, 83.0, 90.0, 76.0, 90.0, 87.0, 93.0, 90.0, 52.0, 88.0, 57.0, 89.0, 97.0, 59.0, 73.0, 38.0, 76.0, 81.0, 45.0, 92.0, 39.0, 34.0, 41.0, 16.0, 33.0, 53.0, 67.0, 57.0, 26.0, 29.0, 10.0, 15.0, 19.0, 10.0, 13.0, 24.0, 20.0, 7.0, 3.0, 16.0, 6.0, 11.0, 8.0, 41.0, 10.0]
    return (endog, exog, labels)