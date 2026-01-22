from mpmath import *
from mpmath.libmp import *
def test_round_nearest():
    assert from_man_exp(0, -4, 4, round_nearest)[:3] == (0, 0, 0)
    assert from_man_exp(240, -4, 4, round_nearest)[:3] == (0, 15, 0)
    assert from_man_exp(247, -4, 4, round_nearest)[:3] == (0, 15, 0)
    assert from_man_exp(248, -4, 4, round_nearest)[:3] == (0, 1, 4)
    assert from_man_exp(249, -4, 4, round_nearest)[:3] == (0, 1, 4)
    assert from_man_exp(232, -4, 4, round_nearest)[:3] == (0, 7, 1)
    assert from_man_exp(233, -4, 4, round_nearest)[:3] == (0, 15, 0)
    assert from_man_exp(-240, -4, 4, round_nearest)[:3] == (1, 15, 0)
    assert from_man_exp(-247, -4, 4, round_nearest)[:3] == (1, 15, 0)
    assert from_man_exp(-248, -4, 4, round_nearest)[:3] == (1, 1, 4)
    assert from_man_exp(-249, -4, 4, round_nearest)[:3] == (1, 1, 4)
    assert from_man_exp(-232, -4, 4, round_nearest)[:3] == (1, 7, 1)
    assert from_man_exp(-233, -4, 4, round_nearest)[:3] == (1, 15, 0)