import collections.abc
import math
import re
import unicodedata
import urllib
from oslo_utils._i18n import _
from oslo_utils import encodeutils
def mask_dict_password(dictionary, secret='***'):
    """Replace password with *secret* in a dictionary recursively.

    :param dictionary: The dictionary which includes secret information.
    :param secret: value with which to replace secret information.
    :returns: The dictionary with string substitutions.

    A dictionary (which may contain nested dictionaries) contains
    information (such as passwords) which should not be revealed, and
    this function helps detect and replace those with the 'secret'
    provided (or `***` if none is provided).

    Substitution is performed in one of three situations:

    If the key is something that is considered to be indicative of a
    secret, then the corresponding value is replaced with the secret
    provided (or `***` if none is provided).

    If a value in the dictionary is a string, then it is masked
    using the ``mask_password()`` function.

    Finally, if a value is a dictionary, this function will
    recursively mask that dictionary as well.

    For example:

    >>> mask_dict_password({'password': 'd81juxmEW_',
    >>>                     'user': 'admin',
    >>>                     'home-dir': '/home/admin'},
    >>>                     '???')
    {'password': '???', 'user': 'admin', 'home-dir': '/home/admin'}

    For example (the value is masked using mask_password())

    >>> mask_dict_password({'password': '--password d81juxmEW_',
    >>>                     'user': 'admin',
    >>>                     'home-dir': '/home/admin'},
    >>>                     '???')
    {'password': '--password ???', 'user': 'admin',
     'home-dir': '/home/admin'}


    For example (a nested dictionary is masked):

    >>> mask_dict_password({"nested": {'password': 'd81juxmEW_',
    >>>                     'user': 'admin',
    >>>                     'home': '/home/admin'}},
    >>>                     '???')
    {"nested": {'password': '???', 'user': 'admin', 'home': '/home/admin'}}

    .. versionadded:: 3.4

    """
    if not isinstance(dictionary, collections.abc.Mapping):
        raise TypeError('Expected a Mapping, got %s instead.' % type(dictionary))
    out = {}
    for k, v in dictionary.items():
        if isinstance(v, collections.abc.Mapping):
            out[k] = mask_dict_password(v, secret=secret)
            continue
        k_matched = False
        if isinstance(k, str):
            for sani_key in _SANITIZE_KEYS:
                if sani_key in k.lower():
                    out[k] = secret
                    k_matched = True
                    break
        if not k_matched:
            if isinstance(v, str):
                out[k] = mask_password(v, secret=secret)
            else:
                out[k] = v
    return out