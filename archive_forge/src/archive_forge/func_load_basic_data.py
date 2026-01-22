from statsmodels.compat.pandas import assert_series_equal, assert_frame_equal
from io import StringIO
from textwrap import dedent
import numpy as np
import numpy.testing as npt
import numpy
from numpy.testing import assert_equal
import pandas
import pytest
from statsmodels.imputation import ros
def load_basic_data():
    raw_csv = StringIO('res,qual\n2.00,=\n4.20,=\n4.62,=\n5.00,ND\n5.00,ND\n5.50,ND\n5.57,=\n5.66,=\n5.75,ND\n5.86,=\n6.65,=\n6.78,=\n6.79,=\n7.50,=\n7.50,=\n7.50,=\n8.63,=\n8.71,=\n8.99,=\n9.50,ND\n9.50,ND\n9.85,=\n10.82,=\n11.00,ND\n11.25,=\n11.25,=\n12.20,=\n14.92,=\n16.77,=\n17.81,=\n19.16,=\n19.19,=\n19.64,=\n20.18,=\n22.97,=\n')
    df = pandas.read_csv(raw_csv)
    df.loc[:, 'conc'] = df['res']
    df.loc[:, 'censored'] = df['qual'] == 'ND'
    return df