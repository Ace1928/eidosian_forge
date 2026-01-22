from . import errors, log
def iter_log_revisions(revisions, revision_source, verbose, rev_tag_dict=None):
    if rev_tag_dict is None:
        rev_tag_dict = {}
    for revno, rev_id, merge_depth in revisions:
        rev = revision_source.get_revision(rev_id)
        if verbose:
            delta = revision_source.get_revision_delta(rev_id)
        else:
            delta = None
        yield log.LogRevision(rev, revno, merge_depth, delta=delta, tags=rev_tag_dict.get(rev_id))