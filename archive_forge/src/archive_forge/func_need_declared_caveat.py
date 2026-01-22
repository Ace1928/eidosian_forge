from ._auth_context import ContextKey
from ._caveat import Caveat, error_caveat, parse_caveat
from ._conditions import (
from ._namespace import Namespace
def need_declared_caveat(cav, keys):
    if cav.location == '':
        return error_caveat('need-declared caveat is not third-party')
    return Caveat(location=cav.location, condition=COND_NEED_DECLARED + ' ' + ','.join(keys) + ' ' + cav.condition)