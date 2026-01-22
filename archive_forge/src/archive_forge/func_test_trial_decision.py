import copy
from time import sleep
import numpy as np
import pandas as pd
from tune.concepts.flow import (
from tune.concepts.space import Rand, TuningParametersTemplate
import cloudpickle
def test_trial_decision():
    trial = Trial('abc', {'a': 1}, {'b': Rand(0, 2)})
    report = TrialReport(trial, metric=np.float(0.1), params={'c': Rand(0, 3)}, metadata={'d': 4})
    decision = TrialDecision(report, budget=0.0, should_checkpoint=True, metadata={'x': 1}, reason='p')
    assert trial is decision.trial
    assert report is decision.report
    decision = cloudpickle.loads(cloudpickle.dumps(decision))
    assert decision.should_stop
    assert decision.should_checkpoint
    assert {'x': 1} == decision.metadata
    assert 'p' == decision.reason
    assert 0.0 == decision.budget
    assert copy.copy(decision) is decision
    assert copy.deepcopy(decision) is decision
    d2 = cloudpickle.loads(cloudpickle.dumps(decision))
    assert d2.trial_id == trial.trial_id
    assert decision.should_stop
    assert decision.should_checkpoint
    assert {'x': 1} == decision.metadata
    assert 'p' == decision.reason
    decision = TrialDecision(report, budget=1.0, should_checkpoint=True, metadata={'x': 1})
    assert 1.0 == decision.budget
    assert not decision.should_stop
    print(decision)