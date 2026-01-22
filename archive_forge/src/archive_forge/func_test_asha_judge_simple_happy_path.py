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
def test_asha_judge_simple_happy_path():
    j = ASHAJudge(schedule=[(1.0, 2), (2.0, 1)])
    d = j.judge(rp('a', 0.5, 0))
    assert 2.0 == d.budget
    assert not d.should_checkpoint
    d = j.judge(rp('b', 0.6, 0))
    assert 2.0 == d.budget
    assert not d.should_checkpoint
    d = j.judge(rp('a', 0.4, 1))
    assert d.should_stop
    assert d.should_checkpoint
    d = j.judge(rp('c', 0.2, 0))
    assert d.should_stop
    assert d.should_checkpoint