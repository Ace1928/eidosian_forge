from concurrent import futures
import weakref
from automaton import machines
from oslo_utils import timeutils
from taskflow import logging
from taskflow import states as st
from taskflow.types import failure
from taskflow.utils import iter_utils
def complete_an_atom(fut):
    atom = fut.atom
    try:
        outcome, result = fut.result()
        do_complete(atom, outcome, result)
        if isinstance(result, failure.Failure):
            retain = do_complete_failure(atom, outcome, result)
            if retain:
                memory.failures.append(result)
            else:
                if LOG.isEnabledFor(logging.DEBUG):
                    intention = get_atom_intention(atom.name)
                    LOG.debug("Discarding failure '%s' (in response to outcome '%s') under completion units request during completion of atom '%s' (intention is to %s)", result, outcome, atom, intention)
                if gather_statistics:
                    statistics['discarded_failures'] += 1
        if gather_statistics:
            statistics['completed'] += 1
    except futures.CancelledError:
        return WAS_CANCELLED
    except Exception:
        memory.failures.append(failure.Failure())
        LOG.exception("Engine '%s' atom post-completion failed", atom)
        return FAILED_COMPLETING
    else:
        return SUCCESSFULLY_COMPLETED