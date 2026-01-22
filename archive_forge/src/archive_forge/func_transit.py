import concurrent.futures as cf
import logging
import sys
from abc import ABC, abstractmethod
from enum import Enum
from threading import Event, RLock
from traceback import StackSummary, extract_stack
from typing import (
from uuid import uuid4
from adagio.exceptions import AbortedError, SkippedError, WorkflowBug
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from six import reraise  # type: ignore
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_instance
from triad.utils.hash import to_uuid
@staticmethod
def transit(state_from: '_State', state_to: '_State') -> '_State':
    if state_from == _State.CREATED:
        if state_to in [_State.RUNNING, _State.ABORTED, _State.SKIPPED, _State.FINISHED]:
            return state_to
    elif state_from == _State.RUNNING:
        if state_to in [_State.FINISHED, _State.ABORTED, _State.FAILED]:
            return state_to
    raise InvalidOperationError(f'Unable to transit from {state_from} to {state_to}')