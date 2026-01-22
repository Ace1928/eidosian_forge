import argparse
import functools
import hashlib
import logging
import os
import socket
import time
import urllib.parse
import warnings
from debtcollector import removals
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import requests
from keystoneclient import exceptions
from keystoneclient.i18n import _
@staticmethod
def register_cli_options(parser):
    """Register the argparse arguments that are needed for a session.

        :param argparse.ArgumentParser parser: parser to add to.
        """
    parser.add_argument('--insecure', default=False, action='store_true', help='Explicitly allow client to perform "insecure" TLS (https) requests. The server\'s certificate will not be verified against any certificate authorities. This option should be used with caution.')
    parser.add_argument('--os-cacert', metavar='<ca-certificate>', default=os.environ.get('OS_CACERT'), help='Specify a CA bundle file to use in verifying a TLS (https) server certificate. Defaults to env[OS_CACERT].')
    parser.add_argument('--os-cert', metavar='<certificate>', default=os.environ.get('OS_CERT'), help='Defaults to env[OS_CERT].')
    parser.add_argument('--os-key', metavar='<key>', default=os.environ.get('OS_KEY'), help='Defaults to env[OS_KEY].')
    parser.add_argument('--timeout', default=600, type=_positive_non_zero_float, metavar='<seconds>', help='Set request timeout (in seconds).')