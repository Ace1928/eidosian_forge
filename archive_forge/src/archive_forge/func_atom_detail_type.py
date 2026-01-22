import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
def atom_detail_type(atom_detail):
    try:
        return _DETAIL_TO_NAME[type(atom_detail)]
    except KeyError:
        raise TypeError("Unknown atom '%s' (%s)" % (atom_detail, type(atom_detail)))