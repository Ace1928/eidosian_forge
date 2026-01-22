from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def find_pseudonyms(repository, revids):
    """Find revisions that are pseudonyms of each other.

    :param repository: Repository object
    :param revids: Sequence of revision ids to check
    :return: Iterable over sets of pseudonyms
    """
    conversions = defaultdict(set)
    conversion_of = defaultdict(set)
    revs = repository.get_revisions(revids)
    pb = ui.ui_factory.nested_progress_bar()
    try:
        for i, rev in enumerate(revs):
            pb.update('finding pseudonyms', i, len(revs))
            for foreign_revid in extract_foreign_revids(rev):
                conversion_of[rev.revision_id].add(foreign_revid)
                conversions[foreign_revid].add(rev.revision_id)
    finally:
        pb.finished()
    done = set()
    for foreign_revid in conversions.keys():
        ret = set()
        check = set(conversions[foreign_revid])
        while check:
            x = check.pop()
            extra = set()
            for frevid in conversion_of[x]:
                extra.update(conversions[frevid])
                del conversions[frevid]
            del conversion_of[x]
            check.update(extra)
            ret.add(x)
        if len(ret) > 1:
            yield ret