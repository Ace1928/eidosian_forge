import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_call_OrdDictEnv():
    ad = rlc.OrdDict(((None, rinterface.parse('sum(x)')),))
    env_a = rinterface.baseenv['new.env']()
    env_a['x'] = rinterface.IntSexpVector([1, 2, 3])
    sum_a = rinterface.baseenv['eval'].rcall(tuple(ad.items()), env_a)
    assert 6 == sum_a[0]
    env_b = rinterface.baseenv['new.env']()
    env_b['x'] = rinterface.IntSexpVector([4, 5, 6])
    sum_b = rinterface.baseenv['eval'].rcall(tuple(ad.items()), env_b)
    assert 15 == sum_b[0]