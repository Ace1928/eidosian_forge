import gzip
import sys
from copy import deepcopy
from os.path import join as pjoin
import numpy as np
import pytest
from .. import csareader as csa
from .. import dwiparams as dwp
from . import dicom_test, pydicom
from .test_dicomwrappers import DATA, IO_DATA_PATH
def test_csas0():
    for csa_str in (CSA2_B0, CSA2_B1000):
        csa_info = csa.read(csa_str)
        assert csa_info['type'] == 2
        assert csa_info['n_tags'] == 83
        tags = csa_info['tags']
        assert len(tags) == 83
        n_o_m = tags['NumberOfImagesInMosaic']
        assert n_o_m['items'] == [48]
    csa_info = csa.read(CSA2_B1000)
    b_matrix = csa_info['tags']['B_matrix']
    assert len(b_matrix['items']) == 6
    b_value = csa_info['tags']['B_value']
    assert b_value['items'] == [1000]