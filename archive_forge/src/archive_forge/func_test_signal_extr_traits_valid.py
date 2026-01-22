import os
import numpy as np
from ...testing import utils
from .. import nilearn as iface
from ...pipeline import engine as pe
import pytest
import numpy.testing as npt
def test_signal_extr_traits_valid(self):
    """Test a node using the SignalExtraction interface.
        Unlike interface.run(), node.run() checks the traits
        """
    node = pe.Node(iface.SignalExtraction(in_file=os.path.abspath(self.filenames['in_file']), label_files=os.path.abspath(self.filenames['label_files']), class_labels=self.labels, incl_shared_variance=False), name='SignalExtraction')
    node.run()