import functools
import logging
import re
from oslo_utils import strutils
from cinderclient._i18n import _
from cinderclient import exceptions
from cinderclient import utils
def get_major_version(self):
    return '%s' % self.ver_major