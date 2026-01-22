import re
import distutils.version
import operator
def split_provision(value):
    """Return the name and optional version number of a provision.

    The version number, if given, will be returned as a `StrictVersion`
    instance, otherwise it will be `None`.

    >>> split_provision('mypkg')
    ('mypkg', None)
    >>> split_provision(' mypkg( 1.2 ) ')
    ('mypkg', StrictVersion ('1.2'))
    """
    global _provision_rx
    if _provision_rx is None:
        _provision_rx = re.compile('([a-zA-Z_]\\w*(?:\\.[a-zA-Z_]\\w*)*)(?:\\s*\\(\\s*([^)\\s]+)\\s*\\))?$', re.ASCII)
    value = value.strip()
    m = _provision_rx.match(value)
    if not m:
        raise ValueError('illegal provides specification: %r' % value)
    ver = m.group(2) or None
    if ver:
        ver = distutils.version.StrictVersion(ver)
    return (m.group(1), ver)