import numpy
import pytest
from thinc.api import Optimizer, registry
def test_optimizer_schedules_invalid(schedule_invalid):
    with pytest.raises(ValueError):
        Optimizer(learn_rate=schedule_invalid)