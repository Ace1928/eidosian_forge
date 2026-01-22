import datetime
import email.utils
import hashlib
import logging
import random
import time
from urllib import parse
from oslo_config import cfg
from swiftclient import client as sc
from swiftclient import exceptions
from swiftclient import utils as swiftclient_utils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
def parse_last_modified(self, lm):
    """Parses the last-modified value.

        For example, last-modified values from a swift object header.
        Returns the datetime.datetime of that value.

        :param lm: The last-modified value (or None)
        :type lm: string
        :returns: An offset-naive UTC datetime of the value (or None)
        """
    if not lm:
        return None
    pd = email.utils.parsedate(lm)[:6]
    return datetime.datetime(*pd)