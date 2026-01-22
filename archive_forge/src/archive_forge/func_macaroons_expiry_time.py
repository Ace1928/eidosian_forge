import pyrfc3339
from ._auth_context import ContextKey
from ._caveat import parse_caveat
from ._conditions import COND_TIME_BEFORE, STD_NAMESPACE
from ._utils import condition_with_prefix
def macaroons_expiry_time(ns, ms):
    """ Returns the minimum time of any time-before caveats found in the given
    macaroons or None if no such caveats were found.
    :param ns: a Namespace, used to resolve caveats.
    :param ms: a list of pymacaroons.Macaroon
    :return: datetime.DateTime or None.
    """
    t = None
    for m in ms:
        et = expiry_time(ns, m.caveats)
        if et is not None and (t is None or et < t):
            t = et
    return t