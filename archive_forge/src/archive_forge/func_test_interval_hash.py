from mpmath import *
def test_interval_hash():
    assert hash(mpi(3)) == hash(3)
    assert hash(mpi(3.25)) == hash(3.25)
    assert hash(mpi(3, 4)) == hash(mpi(3, 4))
    assert hash(iv.mpc(3)) == hash(3)
    assert hash(iv.mpc(3, 4)) == hash(3 + 4j)
    assert hash(iv.mpc((1, 3), (2, 4))) == hash(iv.mpc((1, 3), (2, 4)))