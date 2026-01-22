import unittest
import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
@pytest.mark.parametrize('n_cols, n_rows', [(2, None), (None, 2), (2, 2)])
def test_2x2_lr_bt(self, n_cols, n_rows):
    assert [(0, 0), (100, 0), (0, 100), (100, 100)] == self.compute_layout(n_children=4, ori='lr-bt', n_cols=n_cols, n_rows=n_rows)