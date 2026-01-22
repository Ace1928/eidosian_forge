import abc
from collections import namedtuple
from datetime import datetime
import pyrfc3339
from ._caveat import parse_caveat
from ._conditions import (
from ._declared import DECLARED_KEY
from ._namespace import Namespace
from ._operation import OP_KEY
from ._time import TIME_KEY
from ._utils import condition_with_prefix
def check_first_party_caveat(self, ctx, cav):
    """ Checks the caveat against all registered caveat conditions.
        :return: error message string if any or None
        """
    try:
        cond, arg = parse_caveat(cav)
    except ValueError as ex:
        return 'cannot parse caveat "{}": {}'.format(cav, ex.args[0])
    checker = self._checkers.get(cond)
    if checker is None:
        return 'caveat "{}" not satisfied: caveat not recognized'.format(cav)
    err = checker.check(ctx, cond, arg)
    if err is not None:
        return 'caveat "{}" not satisfied: {}'.format(cav, err)