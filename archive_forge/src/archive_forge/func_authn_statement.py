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
def authn_statement(authn_class=None, authn_auth=None, authn_decl=None, authn_decl_ref=None, authn_instant='', subject_locality='', session_not_on_or_after=None):
    """
    Construct the AuthnStatement
    :param authn_class: Authentication Context Class reference
    :param authn_auth: Authenticating Authority
    :param authn_decl: Authentication Context Declaration
    :param authn_decl_ref: Authentication Context Declaration reference
    :param authn_instant: When the Authentication was performed.
        Assumed to be seconds since the Epoch.
    :param subject_locality: Specifies the DNS domain name and IP address
        for the system from which the assertion subject was apparently
        authenticated.
    :return: An AuthnContext instance
    """
    if authn_instant:
        _instant = instant(time_stamp=authn_instant)
    else:
        _instant = instant()
    if authn_class:
        res = factory(saml.AuthnStatement, authn_instant=_instant, session_index=sid(), session_not_on_or_after=session_not_on_or_after, authn_context=_authn_context_class_ref(authn_class, authn_auth))
    elif authn_decl:
        res = factory(saml.AuthnStatement, authn_instant=_instant, session_index=sid(), session_not_on_or_after=session_not_on_or_after, authn_context=_authn_context_decl(authn_decl, authn_auth))
    elif authn_decl_ref:
        res = factory(saml.AuthnStatement, authn_instant=_instant, session_index=sid(), session_not_on_or_after=session_not_on_or_after, authn_context=_authn_context_decl_ref(authn_decl_ref, authn_auth))
    else:
        res = factory(saml.AuthnStatement, authn_instant=_instant, session_index=sid(), session_not_on_or_after=session_not_on_or_after)
    if subject_locality:
        res.subject_locality = saml.SubjectLocality(text=subject_locality)
    return res