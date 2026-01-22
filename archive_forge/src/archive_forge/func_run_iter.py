import collections
import contextlib
import itertools
import threading
from automaton import runners
from concurrent import futures
import fasteners
import functools
import networkx as nx
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from taskflow.engines.action_engine import builder
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import process_executor
from taskflow.engines.action_engine import runtime
from taskflow.engines import base
from taskflow import exceptions as exc
from taskflow import logging
from taskflow import states
from taskflow import storage
from taskflow.types import failure
from taskflow.utils import misc
def run_iter(self, timeout=None):
    """Runs the engine using iteration (or die trying).

        :param timeout: timeout to wait for any atoms to complete (this timeout
            will be used during the waiting period that occurs after the
            waiting state is yielded when unfinished atoms are being waited
            on).

        Instead of running to completion in a blocking manner, this will
        return a generator which will yield back the various states that the
        engine is going through (and can be used to run multiple engines at
        once using a generator per engine). The iterator returned also
        responds to the ``send()`` method from :pep:`0342` and will attempt to
        suspend itself if a truthy value is sent in (the suspend may be
        delayed until all active atoms have finished).

        NOTE(harlowja): using the ``run_iter`` method will **not** retain the
        engine lock while executing so the user should ensure that there is
        only one entity using a returned engine iterator (one per engine) at a
        given time.
        """
    self.compile()
    self.prepare()
    self.validate()
    last_transitions = collections.deque(maxlen=max(1, self.MAX_MACHINE_STATES_RETAINED))
    with _start_stop(self._task_executor, self._retry_executor):
        self._change_state(states.RUNNING)
        if self._gather_statistics:
            self._statistics.clear()
            w = timeutils.StopWatch()
            w.start()
        else:
            w = None
        try:
            closed = False
            machine, memory = self._runtime.builder.build(self._statistics, timeout=timeout, gather_statistics=self._gather_statistics)
            r = runners.FiniteRunner(machine)
            for transition in r.run_iter(builder.START):
                last_transitions.append(transition)
                _prior_state, new_state = transition
                if new_state in builder.META_STATES:
                    continue
                if new_state == states.FAILURE:
                    failure.Failure.reraise_if_any(memory.failures)
                if closed:
                    continue
                try:
                    try_suspend = (yield new_state)
                except GeneratorExit:
                    closed = True
                    self.suspend()
                except Exception:
                    memory.failures.append(failure.Failure())
                    closed = True
                else:
                    if try_suspend:
                        self.suspend()
        except Exception:
            with excutils.save_and_reraise_exception():
                LOG.exception('Engine execution has failed, something bad must have happened (last %s machine transitions were %s)', last_transitions.maxlen, list(last_transitions))
                self._change_state(states.FAILURE)
        else:
            if last_transitions:
                _prior_state, new_state = last_transitions[-1]
                if new_state not in self.IGNORABLE_STATES:
                    self._change_state(new_state)
                    if new_state not in self.NO_RERAISING_STATES:
                        e_failures = self.storage.get_execute_failures()
                        r_failures = self.storage.get_revert_failures()
                        er_failures = itertools.chain(e_failures.values(), r_failures.values())
                        failure.Failure.reraise_if_any(er_failures)
        finally:
            if w is not None:
                w.stop()
                self._statistics['active_for'] = w.elapsed()