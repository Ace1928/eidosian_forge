from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
def get_hashes(self, extra_data_func=None):
    """Return a dict of computed object hashes.

        :param extra_data_func: a function that is given the object class
                                which gathers more relevant data about the
                                class that is needed in versioning. Returns
                                a tuple containing the extra data bits.
        """
    fingerprints = {}
    for obj_name in sorted(self.obj_classes):
        fingerprints[obj_name] = self._get_fingerprint(obj_name, extra_data_func=extra_data_func)
    return fingerprints