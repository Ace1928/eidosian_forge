import pytest
@pytest.mark.parametrize('n_cols, n_rows', [(4, None), (None, 1), (4, 1)])
@pytest.mark.parametrize('orientation', 'rl-tb rl-bt tb-rl bt-rl'.split())
def test_4x1_rl(self, kivy_clock, orientation, n_cols, n_rows):
    assert {1: (200, 0), 2: (100, 0)} == self.compute_layout(n_data=4, orientation=orientation, n_cols=n_cols, n_rows=n_rows, scroll_to=(150, 0), clock=kivy_clock)