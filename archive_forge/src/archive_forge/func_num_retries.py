import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
@num_retries.setter
def num_retries(self, value):
    util.Typecheck(value, six.integer_types)
    if value < 0:
        raise exceptions.InvalidDataError('Cannot have negative value for num_retries')
    self.__num_retries = value