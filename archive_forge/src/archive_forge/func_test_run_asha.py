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
def test_run_asha(tmpdir):

    class M(Monitor):

        def on_report(self, report: TrialReport) -> None:
            print(report)

    def assert_metric(df: Iterable[Dict[str, Any]], metric: float, ct: int) -> None:
        n = 0
        for row in df:
            assert row[TUNE_REPORT_METRIC] == metric
            n += 1
        assert n == ct
    space = Space(a=Grid(0, 1, 2, 3))
    dag = FugueWorkflow()
    dataset = TuneDatasetBuilder(space, str(tmpdir)).build(dag, shuffle=False)
    obj = F()
    res = optimize_by_continuous_asha(obj, dataset, plan=[[1.0, 3], [1.0, 2], [1.0, 1], [1.0, 1]], checkpoint_path=str(tmpdir))
    res.result(1).output(assert_metric, dict(metric=1.0, ct=1))
    res = optimize_by_continuous_asha(obj, dataset, plan=[[2.0, 2], [1.0, 1], [1.0, 1]], checkpoint_path=str(tmpdir), monitor=M())
    res.result(1).output(assert_metric, dict(metric=1.0, ct=1))
    dag.run()