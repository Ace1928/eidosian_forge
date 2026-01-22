import abc
import datetime
from dogpile.cache import api
from dogpile import util as dp_util
from oslo_cache import core
from oslo_log import log
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_cache._i18n import _
from oslo_cache import exception
def will_copy(self):
    """Will this SON manipulator make a copy of the incoming document?

        Derived classes that do need to make a copy should override this
        method, returning `True` instead of `False`.

        :returns: boolean
        """
    return False