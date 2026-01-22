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
def install_patch(self):
    """
        Install monkeypatch to replace django hasher framework.
        """
    log = self.log
    if self.patched:
        log.warning('monkeypatching already applied, refusing to reapply')
        return False
    if DJANGO_VERSION < MIN_DJANGO_VERSION:
        raise RuntimeError('passlib.ext.django requires django >= %s' % (MIN_DJANGO_VERSION,))
    log.debug('preparing to monkeypatch django ...')
    manager = self._manager
    for record in self.patch_locations:
        if len(record) == 2:
            record += ({},)
        target, source, opts = record
        if target.endswith((':', ',')):
            target += source
        value = getattr(self, source)
        if opts.get('method'):
            value = _wrap_method(value)
        manager.patch(target, value)
    self.reset_hashers()
    self.patched = True
    log.debug('... finished monkeypatching django')
    return True