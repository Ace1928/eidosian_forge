import logging
import re
def max_satisfying(versions, range_, loose=False):
    try:
        range_ob = make_range(range_, loose=loose)
    except Exception:
        return None
    max_ = None
    max_sv = None
    for v in versions:
        if range_ob.test(v):
            if max_ is None or max_sv.compare(v) == -1:
                max_ = v
                max_sv = make_semver(max_, loose=loose)
    return max_