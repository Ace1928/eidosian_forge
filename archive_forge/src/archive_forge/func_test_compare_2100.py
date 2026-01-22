from pandas import Timestamp
def test_compare_2100(self):
    ts = Timestamp('2100-08-12')
    res = ts.to_julian_date()
    assert res == 2488292.5