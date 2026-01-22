from collections import deque, namedtuple
from datetime import timedelta
from celery.utils.functional import memoize
from celery.utils.serialization import strtobool
def find_deprecated_settings(source):
    from celery.utils import deprecated
    for name, opt in flatten(NAMESPACES):
        if (opt.deprecate_by or opt.remove_by) and getattr(source, name, None):
            deprecated.warn(description=f'The {name!r} setting', deprecation=opt.deprecate_by, removal=opt.remove_by, alternative=f'Use the {opt.alt} instead')
    return source