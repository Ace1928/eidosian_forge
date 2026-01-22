from datetime import datetime
from time import sleep
from pytest import raises
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
from tune.noniterative.stopper import (
def test_small_improvement():
    r1 = mr([], 0.5)
    r2 = mr([], 0.4)
    r22 = mr([], 0.51)
    r3 = mr([], 0.39)
    r4 = mr([], 0.2)
    s = small_improvement(0.09, 1)
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert s.can_accept(r3.trial)
    s.judge(r3)
    assert not s.can_accept(r4.trial)
    s = small_improvement(0.25, 2)
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert s.can_accept(r22.trial)
    s.judge(r22)
    assert s.can_accept(r3.trial)
    s.judge(r3)
    assert not s.can_accept(r4.trial)