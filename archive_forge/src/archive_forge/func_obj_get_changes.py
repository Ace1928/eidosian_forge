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
def obj_get_changes(self):
    """Returns a dict of changed fields and their new values."""
    changes = {}
    for key in self.obj_what_changed():
        changes[key] = getattr(self, key)
    return changes