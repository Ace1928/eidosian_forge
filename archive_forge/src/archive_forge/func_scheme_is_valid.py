from . import exceptions
from . import misc
from . import normalizers
def scheme_is_valid(scheme, require=False):
    """Determine if the scheme is valid.

    :param str scheme:
        The scheme string to validate.
    :param bool require:
        (optional) Set to ``True`` to require the presence of a scheme.
    :returns:
        ``True`` if the scheme is valid. ``False`` otherwise.
    :rtype:
        bool
    """
    return is_valid(scheme, misc.SCHEME_MATCHER, require)