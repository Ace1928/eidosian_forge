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
def filter_on_demands(ava, required=None, optional=None):
    """Never return more than is needed. Filters out everything
    the server is prepared to return but the receiver doesn't ask for

    :param ava: Attribute value assertion as a dictionary
    :param required: Required attributes
    :param optional: Optional attributes
    :return: The possibly reduced assertion
    """
    if required is None:
        required = {}
    lava = {k.lower(): k for k in ava.keys()}
    for attr, vals in required.items():
        attr = attr.lower()
        if attr in lava:
            if vals:
                for val in vals:
                    if val not in ava[lava[attr]]:
                        raise MissingValue(f'Required attribute value missing: {attr},{val}')
        else:
            raise MissingValue(f'Required attribute missing: {attr}')
    if optional is None:
        optional = {}
    oka = [k.lower() for k in required.keys()]
    oka.extend([k.lower() for k in optional.keys()])
    for attr in lava.keys():
        if attr not in oka:
            del ava[lava[attr]]
    return ava