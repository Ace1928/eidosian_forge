import sys
import os
import errno
import socket
import warnings
from boto3.exceptions import PythonDeprecationWarning
import collections.abc as collections_abc
def filter_python_deprecation_warnings():
    """
    Invoking this filter acknowledges your runtime will soon be deprecated
    at which time you will stop receiving all updates to your client.
    """
    warnings.filterwarnings('ignore', message='.*Boto3 will no longer support Python.*', category=PythonDeprecationWarning, module='.*boto3\\.compat')