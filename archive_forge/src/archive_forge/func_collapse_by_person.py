import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def collapse_by_person(revisions, canonical_committer):
    """The committers list is sorted by email, fix it up by person.

    Some people commit with a similar username, but different email
    address. Which makes it hard to sort out when they have multiple
    entries. Email is actually more stable, though, since people
    frequently forget to set their name properly.

    So take the most common username for each email address, and
    combine them into one new list.
    """
    committer_to_info = {}
    for rev in revisions:
        authors = rev.get_apparent_authors()
        for author in authors:
            username, email = config.parse_username(author)
            if len(username) == 0 and len(email) == 0:
                continue
            canon_author = canonical_committer[username, email]
            info = committer_to_info.setdefault(canon_author, ([], {}, {}))
            info[0].append(rev)
            info[1][email] = info[1].setdefault(email, 0) + 1
            info[2][username] = info[2].setdefault(username, 0) + 1
    res = [(len(revs), revs, emails, fnames) for revs, emails, fnames in committer_to_info.values()]

    def key_fn(item):
        return (item[0], list(item[2].keys()))
    res.sort(reverse=True, key=key_fn)
    return res