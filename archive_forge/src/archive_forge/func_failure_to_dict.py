import abc
import collections
import threading
from automaton import exceptions as machine_excp
from automaton import machines
import fasteners
import futurist
from oslo_serialization import jsonutils
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.action_engine import executor
from taskflow import exceptions as excp
from taskflow import logging
from taskflow.types import failure as ft
from taskflow.utils import schema_utils as su
def failure_to_dict(failure):
    """Attempts to convert a failure object into a jsonifyable dictionary."""
    failure_dict = failure.to_dict()
    try:
        jsonutils.dumps(failure_dict)
        return failure_dict
    except (TypeError, ValueError):
        return failure.to_dict(include_args=False)