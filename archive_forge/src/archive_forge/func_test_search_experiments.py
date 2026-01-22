import unittest
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import logging as log
@skip_if_no_network
def test_search_experiments(self):
    et = 'xnat:subjectData'
    e = self._intf.array.search_experiments(project_id='ixi', experiment_type=et)
    res = e.data
    self.assertEqual(len(res), 584)