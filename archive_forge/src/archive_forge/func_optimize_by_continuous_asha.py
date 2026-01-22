import os
from typing import Any, Callable, List, Optional, Tuple
from uuid import uuid4
from triad import FileSystem
from tune.api.factory import (
from tune.concepts.dataset import StudyResult, TuneDataset
from tune.concepts.flow import TrialReport
from tune.iterative.asha import ASHAJudge, RungHeap
from tune.iterative.sha import _NonIterativeObjectiveWrapper
from tune.iterative.study import IterativeStudy
from tune.noniterative.study import NonIterativeStudy
def optimize_by_continuous_asha(objective: Any, dataset: TuneDataset, plan: List[Tuple[float, int]], checkpoint_path: str='', always_checkpoint: bool=False, study_early_stop: Optional[Callable[[List[Any], List[RungHeap]], bool]]=None, trial_early_stop: Optional[Callable[[TrialReport, List[TrialReport], List[RungHeap]], bool]]=None, monitor: Any=None) -> StudyResult:
    _objective = parse_iterative_objective(objective)
    _monitor = parse_monitor(monitor)
    checkpoint_path = TUNE_OBJECT_FACTORY.get_path_or_temp(checkpoint_path)
    judge = ASHAJudge(schedule=plan, always_checkpoint=always_checkpoint, study_early_stop=study_early_stop, trial_early_stop=trial_early_stop, monitor=_monitor)
    path = os.path.join(checkpoint_path, str(uuid4()))
    FileSystem().makedirs(path, recreate=True)
    study = IterativeStudy(_objective, checkpoint_path=path)
    return study.optimize(dataset, judge=judge)