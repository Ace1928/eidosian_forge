from datetime import datetime
from time import sleep
from pytest import raises
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
from tune.noniterative.stopper import (
def test_no_update_period():
    r1 = mr([], 0.1)
    r2 = mr([], 0.4)
    r3 = mr([], 0.3)
    r4 = mr([], 0.3)
    s = no_update_period('0.2sec')
    assert s.can_accept(r1.trial)
    s.judge(r1)
    sleep(0.5)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert not s.can_accept(r3.trial)
    s = no_update_period('0.2sec')
    assert s.can_accept(r2.trial)
    s.judge(r2)
    sleep(0.5)
    assert s.can_accept(r3.trial)
    s.judge(r3)
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r4.trial)
    s.judge(r4)