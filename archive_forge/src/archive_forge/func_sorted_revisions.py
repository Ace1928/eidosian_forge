from . import errors, log
def sorted_revisions(revisions, history_map):
    revisions = sorted([(history_map[r], r) for r in revisions])
    return revisions