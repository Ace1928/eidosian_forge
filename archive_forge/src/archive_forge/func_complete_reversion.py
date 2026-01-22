import functools
from taskflow.engines.action_engine.actions import base
from taskflow import logging
from taskflow import states
from taskflow import task as task_atom
from taskflow.types import failure
def complete_reversion(self, task, result):
    if isinstance(result, failure.Failure):
        self.change_state(task, states.REVERT_FAILURE, result=result)
    else:
        self.change_state(task, states.REVERTED, progress=1.0, result=result)