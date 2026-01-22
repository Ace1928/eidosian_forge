import unittest
import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
@pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 1), (3, 1)])
@pytest.mark.parametrize('ori', 'rl-tb rl-bt tb-rl bt-rl'.split())
def test_3x1_rl(self, ori, n_cols, n_rows):
    assert [(200, 0), (100, 0), (0, 0)] == self.compute_layout(n_children=3, ori=ori, n_cols=n_cols, n_rows=n_rows)