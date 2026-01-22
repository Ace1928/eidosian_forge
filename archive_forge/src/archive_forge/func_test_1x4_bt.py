import pytest
@pytest.mark.parametrize('n_cols, n_rows', [(1, None), (None, 4), (1, 4)])
@pytest.mark.parametrize('orientation', 'bt-lr bt-rl lr-bt rl-bt'.split())
def test_1x4_bt(self, kivy_clock, orientation, n_cols, n_rows):
    assert {1: (0, 100), 2: (0, 200)} == self.compute_layout(n_data=4, orientation=orientation, n_cols=n_cols, n_rows=n_rows, scroll_to=(0, 150), clock=kivy_clock)