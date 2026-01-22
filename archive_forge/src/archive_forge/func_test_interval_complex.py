from mpmath import *
def test_interval_complex():
    iv.dps = 15
    mp.dps = 15
    assert iv.mpc(2, 3) == 2 + 3j
    assert iv.mpc(2, 3) != 2 + 4j
    assert iv.mpc(2, 3) != 1 + 3j
    assert 1 + 3j in iv.mpc([1, 2], [3, 4])
    assert 2 + 5j not in iv.mpc([1, 2], [3, 4])
    assert iv.mpc(1, 2) + 1j == 1 + 3j
    assert iv.mpc([1, 2], [2, 3]) + 2 + 3j == iv.mpc([3, 4], [5, 6])
    assert iv.mpc([2, 4], [4, 8]) / 2 == iv.mpc([1, 2], [2, 4])
    assert iv.mpc([1, 2], [2, 4]) * 2j == iv.mpc([-8, -4], [2, 4])
    assert iv.mpc([2, 4], [4, 8]) / 2j == iv.mpc([2, 4], [-2, -1])
    assert iv.exp(2 + 3j).ae(mp.exp(2 + 3j))
    assert iv.log(2 + 3j).ae(mp.log(2 + 3j))
    assert (iv.mpc(2, 3) ** iv.mpc(0.5, 2)).ae(mp.mpc(2, 3) ** mp.mpc(0.5, 2))
    assert 1j in iv.mpf(-1) ** 0.5
    assert 1j in iv.mpc(-1) ** 0.5
    assert abs(iv.mpc(0)) == 0
    assert abs(iv.mpc(inf)) == inf
    assert abs(iv.mpc(3, 4)) == 5
    assert abs(iv.mpc(4)) == 4
    assert abs(iv.mpc(0, 4)) == 4
    assert abs(iv.mpc(0, [2, 3])) == iv.mpf([2, 3])
    assert abs(iv.mpc(0, [-3, 2])) == iv.mpf([0, 3])
    assert abs(iv.mpc([3, 5], [4, 12])) == iv.mpf([5, 13])
    assert abs(iv.mpc([3, 5], [-4, 12])) == iv.mpf([3, 13])
    assert iv.mpc(2, 3) ** 0 == 1
    assert iv.mpc(2, 3) ** 1 == 2 + 3j
    assert iv.mpc(2, 3) ** 2 == (2 + 3j) ** 2
    assert iv.mpc(2, 3) ** 3 == (2 + 3j) ** 3
    assert iv.mpc(2, 3) ** 4 == (2 + 3j) ** 4
    assert iv.mpc(2, 3) ** 5 == (2 + 3j) ** 5
    assert iv.mpc(2, 2) ** (-1) == (2 + 2j) ** (-1)
    assert iv.mpc(2, 2) ** (-2) == (2 + 2j) ** (-2)
    assert iv.cos(2).ae(mp.cos(2))
    assert iv.sin(2).ae(mp.sin(2))
    assert iv.cos(2 + 3j).ae(mp.cos(2 + 3j))
    assert iv.sin(2 + 3j).ae(mp.sin(2 + 3j))