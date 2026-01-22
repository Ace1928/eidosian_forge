import abc
import copy
import functools
import urllib
import warnings
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import plugin
from oslo_utils import strutils
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.i18n import _
def filter_kwargs(f):

    @functools.wraps(f)
    def func(*args, **kwargs):
        new_kwargs = {}
        for key, ref in kwargs.items():
            if ref is None:
                continue
            id_value = getid(ref)
            if id_value != ref:
                key = '%s_id' % key
            new_kwargs[key] = id_value
        return f(*args, **new_kwargs)
    return func