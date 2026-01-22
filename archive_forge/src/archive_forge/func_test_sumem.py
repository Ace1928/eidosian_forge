from mpmath import *
def test_sumem():
    mp.dps = 15
    assert sumem(lambda k: 1 / k ** 2.5, [50, 100]).ae(0.0012524505324784962)
    assert sumem(lambda k: k ** 4 + 3 * k + 1, [10, 100]).ae(2050333103)