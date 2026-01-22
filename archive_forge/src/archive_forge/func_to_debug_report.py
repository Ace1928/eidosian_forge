from __future__ import absolute_import
import urllib3
import copy
import logging
import multiprocessing
import sys
from six import iteritems
from six import with_metaclass
from six.moves import http_client as httplib
def to_debug_report(self):
    """
        Gets the essential information for debugging.

        :return: The report for debugging.
        """
    return 'Python SDK Debug Report:\nOS: {env}\nPython Version: {pyversion}\nVersion of the API: v1.14.4\nSDK Package Version: 10.0.0-snapshot'.format(env=sys.platform, pyversion=sys.version)