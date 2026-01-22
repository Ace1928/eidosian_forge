import os
import numpy as np
from ...testing import utils
from .. import nilearn as iface
from ...pipeline import engine as pe
import pytest
import numpy.testing as npt
def test_signal_extr_bad_label_list(self):
    with pytest.raises(ValueError):
        iface.SignalExtraction(in_file=self.filenames['in_file'], label_files=self.filenames['label_files'], class_labels=['bad'], incl_shared_variance=False).run()