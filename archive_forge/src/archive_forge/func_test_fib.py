def test_fib(self):
    """Test the fibonacci sequence generator"""
    from boto.sdb.db.sequence import fib
    lv = 0
    for v in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]:
        assert fib(v, lv) == lv + v
        lv = fib(v, lv)