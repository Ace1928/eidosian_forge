import time
import joblib
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import config_context, get_config
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.parallel import Parallel, delayed
def get_working_memory():
    return get_config()['working_memory']