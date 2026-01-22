import numpy
import pytest
from thinc.api import Optimizer, registry
def test_optimizer_schedules_from_config(schedule_valid):
    lr, lr_next1, lr_next2, lr_next3 = schedule_valid
    cfg = {'@optimizers': 'Adam.v1', 'learn_rate': lr}
    optimizer = registry.resolve({'cfg': cfg})['cfg']
    assert optimizer.learn_rate == lr_next1
    optimizer.step_schedules()
    assert optimizer.learn_rate == lr_next2
    optimizer.step_schedules()
    assert optimizer.learn_rate == lr_next3
    optimizer.learn_rate = 1.0
    assert optimizer.learn_rate == 1.0