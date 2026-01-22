import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
def test_lasso_lars_vs_R_implementation():
    y = np.array([-6.45006793, -3.51251449, -8.52445396, 6.12277822, -19.42109366])
    x = np.array([[0.47299829, 0, 0, 0, 0], [0.08239882, 0.85784863, 0, 0, 0], [0.30114139, -0.07501577, 0.80895216, 0, 0], [-0.01460346, -0.1015233, 0.0407278, 0.80338378, 0], [-0.69363927, 0.06754067, 0.18064514, -0.0803561, 0.40427291]])
    X = x.T
    r = np.array([[0, 0, 0, 0, 0, -79.81036280949903, -83.52878873278283, -83.77765373919071, -83.78415693288893, -84.03339059175666], [0, 0, 0, 0, -0.476624256777266, 0, 0, 0, 0, 0.025219751009936], [0, -3.577397088285891, -4.702795355871871, -7.016748621359461, -7.614898471899412, -0.336938391359179, 0, 0, 0.001213370600853, 0.048162321585148], [0, 0, 0, 2.231558436628169, 2.723267514525966, 2.811549786389614, 2.813766976061531, 2.817462468949557, 2.817368178703816, 2.816221090636795], [0, 0, -1.218422599914637, -3.457726183014808, -4.02130452206071, -45.827461592423745, -47.776608869312305, -47.9115616107464, -47.914845922736234, -48.03956233426572]])
    model_lasso_lars = linear_model.LassoLars(alpha=0, fit_intercept=False)
    model_lasso_lars.fit(X, y)
    skl_betas = model_lasso_lars.coef_path_
    assert_array_almost_equal(r, skl_betas, decimal=12)