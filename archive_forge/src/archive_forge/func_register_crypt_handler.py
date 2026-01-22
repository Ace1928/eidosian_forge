import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedTypeError, PasslibWarning
from passlib.ifc import PasswordHash
from passlib.utils import (
from passlib.utils.compat import unicode_or_str
from passlib.utils.decor import memoize_single_value
def register_crypt_handler(handler, force=False, _attr=None):
    """register password hash handler.

    this method immediately registers a handler with the internal passlib registry,
    so that it will be returned by :func:`get_crypt_handler` when requested.

    :arg handler: the password hash handler to register
    :param force: force override of existing handler (defaults to False)
    :param _attr:
        [internal kwd] if specified, ensures ``handler.name``
        matches this value, or raises :exc:`ValueError`.

    :raises TypeError:
        if the specified object does not appear to be a valid handler.

    :raises ValueError:
        if the specified object's name (or other required attributes)
        contain invalid values.

    :raises KeyError:
        if a (different) handler was already registered with
        the same name, and ``force=True`` was not specified.
    """
    if not is_crypt_handler(handler):
        raise ExpectedTypeError(handler, 'password hash handler', 'handler')
    if not handler:
        raise AssertionError('``bool(handler)`` must be True')
    name = handler.name
    _validate_handler_name(name)
    if _attr and _attr != name:
        raise ValueError('handlers must be stored only under their own name (%r != %r)' % (_attr, name))
    other = _handlers.get(name)
    if other:
        if other is handler:
            log.debug('same %r handler already registered: %r', name, handler)
            return
        elif force:
            log.warning('overriding previously registered %r handler: %r', name, other)
        else:
            raise KeyError('another %r handler has already been registered: %r' % (name, other))
    _handlers[name] = handler
    log.debug('registered %r handler: %r', name, handler)