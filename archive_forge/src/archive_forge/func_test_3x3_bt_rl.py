import pytest
@pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 3), (3, 3)])
def test_3x3_bt_rl(self, kivy_clock, n_cols, n_rows):
    assert {4: (100, 100), 5: (100, 200), 7: (0, 100)} == self.compute_layout(n_data=8, orientation='bt-rl', n_cols=n_cols, n_rows=n_rows, scroll_to=(50, 150), clock=kivy_clock)