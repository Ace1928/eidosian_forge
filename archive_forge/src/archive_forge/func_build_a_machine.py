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
def build_a_machine(freeze=True):
    """Builds a state machine that requests are allowed to go through."""
    m = machines.FiniteMachine()
    for st in (WAITING, PENDING, RUNNING):
        m.add_state(st)
    for st in (SUCCESS, FAILURE):
        m.add_state(st, terminal=True)
    m.default_start_state = WAITING
    m.add_transition(WAITING, PENDING, make_an_event(PENDING))
    m.add_transition(WAITING, FAILURE, make_an_event(FAILURE))
    m.add_transition(PENDING, RUNNING, make_an_event(RUNNING))
    m.add_transition(PENDING, FAILURE, make_an_event(FAILURE))
    m.add_transition(RUNNING, FAILURE, make_an_event(FAILURE))
    m.add_transition(RUNNING, SUCCESS, make_an_event(SUCCESS))
    if freeze:
        m.freeze()
    return m