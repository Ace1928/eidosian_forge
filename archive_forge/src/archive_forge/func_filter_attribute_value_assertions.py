import copy
import importlib
import logging
import re
from warnings import warn as _warn
from saml2 import saml
from saml2 import xmlenc
from saml2.attribute_converter import ac_factory
from saml2.attribute_converter import from_local
from saml2.attribute_converter import get_local_name
from saml2.s_utils import MissingValue
from saml2.s_utils import assertion_factory
from saml2.s_utils import factory
from saml2.s_utils import sid
from saml2.saml import NAME_FORMAT_URI
from saml2.time_util import in_a_while
from saml2.time_util import instant
def filter_attribute_value_assertions(ava, attribute_restrictions=None):
    """Will weed out attribute values and values according to the
    rules defined in the attribute restrictions. If filtering results in
    an attribute without values, then the attribute is removed from the
    assertion.

    :param ava: The incoming attribute value assertion (dictionary)
    :param attribute_restrictions: The rules that govern which attributes
        and values that are allowed. (dictionary)
    :return: The modified attribute value assertion
    """
    if not attribute_restrictions:
        return ava
    for attr, vals in list(ava.items()):
        _attr = attr.lower()
        try:
            _rests = attribute_restrictions[_attr]
        except KeyError:
            del ava[attr]
        else:
            if _rests is None:
                continue
            if isinstance(vals, str):
                vals = [vals]
            rvals = []
            for restr in _rests:
                for val in vals:
                    if restr.match(val):
                        rvals.append(val)
            if rvals:
                ava[attr] = list(set(rvals))
            else:
                del ava[attr]
    return ava