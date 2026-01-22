import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_survdiff():
    full_df = bmt.copy()
    df = bmt[bmt.Group != 'ALL'].copy()
    stat, p = survdiff(df['T'], df.Status, df.Group)
    assert_allclose(stat, 13.44556, atol=0.0001, rtol=0.0001)
    stat, p = survdiff(df['T'], df.Status, df.Group, weight_type='gb')
    assert_allclose(stat, 15.38787, atol=0.0001, rtol=0.0001)
    stat, p = survdiff(df['T'], df.Status, df.Group, weight_type='tw')
    assert_allclose(stat, 14.98382, atol=0.0001, rtol=0.0001)
    stat, p = survdiff(df['T'], df.Status, df.Group, weight_type='fh', fh_p=0.5)
    assert_allclose(stat, 14.46866, atol=0.0001, rtol=0.0001)
    stat, p = survdiff(df['T'], df.Status, df.Group, weight_type='fh', fh_p=1)
    assert_allclose(stat, 14.845, atol=0.0001, rtol=0.0001)
    stat, p = survdiff(full_df['T'], full_df.Status, full_df.Group, weight_type='fh', fh_p=1)
    assert_allclose(stat, 15.67247, atol=0.0001, rtol=0.0001)
    strata = np.arange(df.shape[0]) % 5
    df['strata'] = strata
    stat, p = survdiff(df['T'], df.Status, df.Group, strata=df.strata)
    assert_allclose(stat, 11.97799, atol=0.0001, rtol=0.0001)
    stat, p = survdiff(df['T'], df.Status, df.Group, strata=df.strata, weight_type='fh', fh_p=0.5)
    assert_allclose(stat, 12.6257, atol=0.0001, rtol=0.0001)
    stat, p = survdiff(df['T'], df.Status, df.Group, strata=df.strata, weight_type='fh', fh_p=1)
    assert_allclose(stat, 12.73565, atol=0.0001, rtol=0.0001)
    full_strata = np.arange(full_df.shape[0]) % 5
    full_df['strata'] = full_strata
    stat, p = survdiff(full_df['T'], full_df.Status, full_df.Group, strata=full_df.strata, weight_type='fh', fh_p=0.5)
    assert_allclose(stat, 13.56793, atol=0.0001, rtol=0.0001)
    df['strata'] = np.arange(df.shape[0]) % 8
    stat, p = survdiff(df['T'], df.Status, df.Group, strata=df.strata)
    assert_allclose(stat, 12.12631, atol=0.0001, rtol=0.0001)
    stat, p = survdiff(df['T'], df.Status, df.Group, strata=df.strata, weight_type='fh', fh_p=0.5)
    assert_allclose(stat, 12.9633, atol=0.0001, rtol=0.0001)
    stat, p = survdiff(df['T'], df.Status, df.Group, strata=df.strata, weight_type='fh', fh_p=1)
    assert_allclose(stat, 13.35259, atol=0.0001, rtol=0.0001)