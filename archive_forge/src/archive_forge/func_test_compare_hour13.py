from pandas import Timestamp
def test_compare_hour13(self):
    ts = Timestamp('2000-08-12T13:00:00')
    res = ts.to_julian_date()
    assert res == 2451769.0416666665