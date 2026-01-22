import unittest
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import logging as log
@skip_if_no_network
def test_array_experiments(self):
    """
        Get a list of experiments from a given subject which has multiple types
        of experiments (i.e. MRSessions and PETSessions) and assert it gathers
        them all.
        """
    e = self._intf.array.experiments(subject_id='XNAT_S04207').data
    self.assertEqual(len(e), 2)