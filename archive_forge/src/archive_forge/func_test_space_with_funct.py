from tune import Choice, Grid, Rand, RandInt, Space, TuningParametersTemplate, FuncParam
from pytest import raises
def test_space_with_funct():
    s = Space(a=1, b=FuncParam(lambda x, y: x + y, x=Grid(0, 1), y=Grid(3, 4)))
    assert [dict(a=1, b=3), dict(a=1, b=4), dict(a=1, b=4), dict(a=1, b=5)] == list(s)
    u = Grid(0, 1)
    s = Space(a=u, b=FuncParam(lambda x, y: x + y, x=u, y=u))
    assert [dict(a=0, b=0), dict(a=1, b=2)] == list(s)