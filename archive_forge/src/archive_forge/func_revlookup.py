from . import Base
from . Base import ServerError
def revlookup(name, timeout=30):
    """convenience routine for doing a reverse lookup of an address"""
    if Base.defaults['server'] == []:
        Base.DiscoverNameServers()
    names = revlookupall(name, timeout)
    if not names:
        return None
    return names[0]