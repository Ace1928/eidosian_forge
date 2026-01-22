import getpass
import inspect
import os
import sys
import textwrap
import decorator
from magnumclient.common.apiclient import exceptions
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from magnumclient.i18n import _
def deprecation_message(preamble, new_name):
    msg = '%s This parameter is deprecated and will be removed in a future release. Use --%s instead.' % (preamble, new_name)
    return msg