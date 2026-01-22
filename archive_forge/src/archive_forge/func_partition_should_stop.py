from datetime import datetime
from time import sleep
from pytest import raises
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
from tune.noniterative.stopper import (
def partition_should_stop(self, latest_report, updated, reports) -> bool:
    self._last = latest_report
    return self._func(latest_report, updated, reports)