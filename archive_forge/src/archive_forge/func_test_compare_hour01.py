from pandas import Timestamp
def test_compare_hour01(self):
    ts = Timestamp('2000-08-12T01:00:00')
    res = ts.to_julian_date()
    assert res == 2451768.5416666665