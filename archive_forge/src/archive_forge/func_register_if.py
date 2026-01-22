import abc
import collections
from collections import abc as collections_abc
import copy
import functools
import logging
import warnings
import oslo_messaging as messaging
from oslo_utils import excutils
from oslo_utils import versionutils as vutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields as obj_fields
@classmethod
def register_if(cls, condition):

    def wraps(obj_cls):
        if condition:
            obj_cls = cls.register(obj_cls)
        else:
            _make_class_properties(obj_cls)
        return obj_cls
    return wraps