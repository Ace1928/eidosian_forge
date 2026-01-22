from __future__ import (absolute_import, division, print_function)
def sorted_ttls(ttls):
    return sorted(ttls, key=lambda ttl: 0 if ttl is None else ttl)