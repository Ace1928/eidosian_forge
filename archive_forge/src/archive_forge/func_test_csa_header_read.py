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
@dicom_test
def test_csa_header_read():
    hdr = csa.get_csa_header(DATA, 'image')
    assert hdr['n_tags'] == 83
    assert csa.get_csa_header(DATA, 'series')['n_tags'] == 65
    with pytest.raises(ValueError):
        csa.get_csa_header(DATA, 'xxxx')
    assert csa.is_mosaic(hdr)
    data2 = pydicom.dataset.Dataset()
    for element in DATA:
        if (element.tag.group, element.tag.elem) != (41, 16):
            data2.add(element)
    assert csa.get_csa_header(data2, 'image') is None
    data2[41, 16] = DATA[41, 16]
    assert csa.is_mosaic(csa.get_csa_header(data2, 'image'))