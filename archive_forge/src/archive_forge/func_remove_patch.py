from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
def remove_patch(self):
    """
        Remove monkeypatch from django hasher framework.
        As precaution in case there are lingering refs to context,
        context object will be wiped.

        .. warning::
            This may cause problems if any other Django modules have imported
            their own copies of the patched functions, though the patched
            code has been designed to throw an error as soon as possible in
            this case.
        """
    log = self.log
    manager = self._manager
    if self.patched:
        log.debug('removing django monkeypatching...')
        manager.unpatch_all(unpatch_conflicts=True)
        self.context.load({})
        self.patched = False
        self.reset_hashers()
        log.debug('...finished removing django monkeypatching')
        return True
    if manager.isactive():
        log.warning('reverting partial monkeypatching of django...')
        manager.unpatch_all()
        self.context.load({})
        self.reset_hashers()
        log.debug('...finished removing django monkeypatching')
        return True
    log.debug('django not monkeypatched')
    return False