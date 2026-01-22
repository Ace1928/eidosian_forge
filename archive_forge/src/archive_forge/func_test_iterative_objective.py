import pandas as pd
from fugue.workflow.workflow import FugueWorkflow
from pytest import raises
from tune import Space, MetricLogger
from tune.api.factory import (
from tune.concepts.dataset import TuneDataset
from tune.concepts.flow.judge import Monitor
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
from tune_optuna.optimizer import OptunaLocalOptimizer
def test_iterative_objective():

    class Mock(IterativeObjectiveFunc):
        pass
    assert isinstance(parse_iterative_objective(Mock()), IterativeObjectiveFunc)
    with raises(TuneCompileError):
        parse_iterative_objective('x')

    @parse_iterative_objective.candidate(lambda obj: isinstance(obj, _Dummy))
    def _converter(obj):
        return Mock()
    assert isinstance(parse_iterative_objective(_DUMMY), IterativeObjectiveFunc)