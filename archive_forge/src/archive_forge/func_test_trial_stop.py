import math
from typing import Any, Dict, Iterable
from fugue import FugueWorkflow
from tune import optimize_by_continuous_asha
from tune.constants import TUNE_REPORT_METRIC
from tune.concepts.dataset import TuneDatasetBuilder
from tune.iterative.asha import ASHAJudge, RungHeap
from tune.iterative.objective import IterativeObjectiveFunc
from tune.concepts.space import Grid, Space
from tune.concepts.flow import Monitor, Trial, TrialReport
def test_trial_stop():

    def should_stop(report, history, rungs):
        return not all((report.trial_id in x for x in rungs[:report.rung]))
    j = ASHAJudge(schedule=[(1.0, 2), (2.0, 2), (3.0, 1)], always_checkpoint=True, trial_early_stop=should_stop)
    d = j.judge(rp('a', 0.6, 0))
    assert 2.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp('b', 0.5, 0))
    assert 2.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp('c', 0.4, 0))
    assert 2.0 == d.budget
    assert d.should_checkpoint
    d = j.judge(rp('a', 0.1, 1))
    assert d.should_stop
    assert d.should_checkpoint