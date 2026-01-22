import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def gather_class_stats(repository, revs):
    ret = {}
    total = 0
    with ui.ui_factory.nested_progress_bar() as pb:
        with repository.lock_read():
            i = 0
            for delta in repository.get_revision_deltas(revs):
                pb.update('classifying commits', i, len(revs))
                for c in classify_delta(delta):
                    if c not in ret:
                        ret[c] = 0
                    ret[c] += 1
                    total += 1
                i += 1
    return (ret, total)