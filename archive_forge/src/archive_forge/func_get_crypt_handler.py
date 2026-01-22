import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedTypeError, PasslibWarning
from passlib.ifc import PasswordHash
from passlib.utils import (
from passlib.utils.compat import unicode_or_str
from passlib.utils.decor import memoize_single_value
def get_crypt_handler(name, default=_UNSET):
    """return handler for specified password hash scheme.

    this method looks up a handler for the specified scheme.
    if the handler is not already loaded,
    it checks if the location is known, and loads it first.

    :arg name: name of handler to return
    :param default: optional default value to return if no handler with specified name is found.

    :raises KeyError: if no handler matching that name is found, and no default specified, a KeyError will be raised.

    :returns: handler attached to name, or default value (if specified).
    """
    if name.startswith('_'):
        if default is _UNSET:
            raise KeyError('invalid handler name: %r' % (name,))
        else:
            return default
    try:
        return _handlers[name]
    except KeyError:
        pass
    assert isinstance(name, unicode_or_str), 'name must be string instance'
    alt = name.replace('-', '_').lower()
    if alt != name:
        warn('handler names should be lower-case, and use underscores instead of hyphens: %r => %r' % (name, alt), PasslibWarning, stacklevel=2)
        name = alt
        try:
            return _handlers[name]
        except KeyError:
            pass
    path = _locations.get(name)
    if path:
        if ':' in path:
            modname, modattr = path.split(':')
        else:
            modname, modattr = (path, name)
        mod = __import__(modname, fromlist=[modattr], level=0)
        handler = _handlers.get(name)
        if handler:
            assert is_crypt_handler(handler), 'unexpected object: name=%r object=%r' % (name, handler)
            return handler
        handler = getattr(mod, modattr)
        register_crypt_handler(handler, _attr=name)
        return handler
    if default is _UNSET:
        raise KeyError('no crypt handler found for algorithm: %r' % (name,))
    else:
        return default