import os
import unittest
from unittest import mock
import numpy as np
import pytest
import nibabel as nb
from nibabel.cmdline.roi import lossless_slice, main, parse_slice
from nibabel.testing import data_path
@pytest.mark.parametrize('args, errmsg', ((('-i', '1:1'), 'Cannot take zero-length slice'), (('-j', '1::2'), 'Downsampling is not supported'), (('-t', '5::-1'), 'Step entry not permitted')))
def test_nib_roi_bad_slices(capsys, args, errmsg):
    in_file = os.path.join(data_path, 'functional.nii')
    retval = main([in_file, os.devnull, *args])
    assert retval != 0
    captured = capsys.readouterr()
    assert errmsg in captured.out