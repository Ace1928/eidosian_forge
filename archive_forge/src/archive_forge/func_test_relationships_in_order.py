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
def test_relationships_in_order(self):
    for obj_name in self.obj_classes:
        obj_classes = self.obj_classes[obj_name]
        for obj_class in obj_classes:
            self._test_relationships_in_order(obj_class)