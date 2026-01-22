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
def get_dependency_tree(self):
    tree = {}
    for obj_name in self.obj_classes.keys():
        self._get_dependencies(tree, self.obj_classes[obj_name][0])
    return tree