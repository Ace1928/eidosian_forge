from datetime import datetime
from time import sleep
from pytest import raises
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
from tune.noniterative.stopper import (
def test_logical_ops():
    r1 = mr([], 0.5)
    r2 = mr([], 0.4)
    r3 = mr([], 0.3)
    r4 = mr([], 0.2)
    take_two = MockSimpleStopper(lambda latest, updated, reports: len(reports) >= 2)
    ends_small = MockSimpleStopper(lambda latest, updated, reports: reports[-1].sort_metric <= 0.3)
    s = take_two & ends_small
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert s.can_accept(r3.trial)
    s.judge(r3)
    assert not s.can_accept(r4.trial)
    with raises(AssertionError):
        take_two | ends_small
    take_two = MockSimpleStopper(lambda latest, updated, reports: len(reports) >= 2)
    ends_small = MockSimpleStopper(lambda latest, updated, reports: reports[-1].sort_metric <= 0.3)
    s = take_two | ends_small
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert not s.can_accept(r3.trial)