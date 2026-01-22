from io import StringIO
from os.path import join as pjoin
import numpy as np
import pytest
import nibabel as nib
from nibabel.cmdline.diff import *
from nibabel.cmdline.utils import *
from nibabel.testing import data_path
def test_get_data_diff():
    test_names = [pjoin(data_path, f) for f in ('standard.nii.gz', 'standard.nii.gz')]
    assert get_data_hash_diff(test_names) == []
    test_array = np.arange(16).reshape(4, 4)
    test_array_2 = np.arange(1, 17).reshape(4, 4)
    test_array_3 = np.arange(2, 18).reshape(4, 4)
    test_array_4 = np.arange(100).reshape(10, 10)
    test_array_5 = np.arange(64).reshape(8, 8)
    assert get_data_diff([test_array, test_array_2]) == {'DATA(diff 1:)': [None, {'abs': 1, 'rel': 2.0}]}
    assert get_data_diff([test_array, test_array_2, test_array_3]) == {'DATA(diff 1:)': [None, {'abs': 1, 'rel': 2.0}, {'abs': 2, 'rel': 2.0}], 'DATA(diff 2:)': [None, None, {'abs': 1, 'rel': 0.6666666666666666}]}
    assert get_data_diff([test_array, test_array_2], max_abs=2, max_rel=2) == {}
    assert get_data_diff([test_array_2, test_array_4]) == {'DATA(diff 1:)': [None, {'CMP': 'incompat'}]}
    assert get_data_diff([test_array_4, test_array_5, test_array_2]) == {'DATA(diff 1:)': [None, {'CMP': 'incompat'}, {'CMP': 'incompat'}], 'DATA(diff 2:)': [None, None, {'CMP': 'incompat'}]}
    test_return = get_data_diff([test_array, test_array_2], dtype=np.float32)
    assert type(test_return['DATA(diff 1:)'][1]['abs']) is np.float32
    assert type(test_return['DATA(diff 1:)'][1]['rel']) is np.float32
    test_return_2 = get_data_diff([test_array, test_array_2, test_array_3])
    assert type(test_return_2['DATA(diff 1:)'][1]['abs']) is np.float64
    assert type(test_return_2['DATA(diff 1:)'][1]['rel']) is np.float64
    assert type(test_return_2['DATA(diff 2:)'][2]['abs']) is np.float64
    assert type(test_return_2['DATA(diff 2:)'][2]['rel']) is np.float64