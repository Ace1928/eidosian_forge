from collections.abc import Mapping
import copy
import logging
import sys
import traceback
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging import _utils as utils
def deserialize_remote_exception(data, allowed_remote_exmods):
    failure = jsonutils.loads(str(data))
    trace = failure.get('tb', [])
    message = failure.get('message', '') + '\n' + '\n'.join(trace)
    name = failure.get('class')
    module = failure.get('module')
    if module in _EXCEPTIONS_MODULES:
        module = _EXCEPTIONS_MODULE
    if module != _EXCEPTIONS_MODULE and module not in allowed_remote_exmods:
        return oslo_messaging.RemoteError(name, failure.get('message'), trace)
    try:
        __import__(module)
        mod = sys.modules[module]
        klass = getattr(mod, name)
        if not issubclass(klass, Exception):
            raise TypeError('Can only deserialize Exceptions')
        failure = klass(*failure.get('args', []), **failure.get('kwargs', {}))
    except (AttributeError, TypeError, ImportError) as error:
        LOG.warning('Failed to rebuild remote exception due to error: %s', str(error))
        return oslo_messaging.RemoteError(name, failure.get('message'), trace)
    ex_type = type(failure)
    str_override = lambda self: message
    new_ex_type = type(ex_type.__name__ + _REMOTE_POSTFIX, (ex_type,), {'__str__': str_override, '__unicode__': str_override})
    new_ex_type.__module__ = '%s%s' % (module, _REMOTE_POSTFIX)
    try:
        failure.__class__ = new_ex_type
    except TypeError:
        failure.args = (message,) + failure.args[1:]
    return failure