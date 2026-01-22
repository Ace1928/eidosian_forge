from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
def test_combine_indices():
    np.random.seed(985367)
    groups = np.random.randint(0, 2, size=(10, 2))
    uv, ux, u, label = combine_indices(groups, return_labels=True)
    uv, ux, u, label = combine_indices(groups, prefix='g1,g2=', sep=',', return_labels=True)
    group0 = np.array(['sector0', 'sector1'])[groups[:, 0]]
    group1 = np.array(['region0', 'region1'])[groups[:, 1]]
    uv, ux, u, label = combine_indices((group0, group1), prefix='sector,region=', sep=',', return_labels=True)
    uv, ux, u, label = combine_indices((group0, group1), prefix='', sep='.', return_labels=True)
    group_joint = np.array(label)[uv.flat]
    group_joint_expected = np.array(['sector1.region0', 'sector0.region1', 'sector0.region0', 'sector0.region1', 'sector1.region1', 'sector0.region0', 'sector1.region0', 'sector1.region0', 'sector0.region1', 'sector0.region0'], dtype='|U15')
    assert_equal(group_joint, group_joint_expected)