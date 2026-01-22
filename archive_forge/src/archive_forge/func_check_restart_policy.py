import base64
import binascii
import os
import re
import shlex
from oslo_serialization import jsonutils
from oslo_utils import netutils
from urllib import parse
from urllib import request
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import cliutils as utils
from zunclient import exceptions as exc
from zunclient.i18n import _
def check_restart_policy(policy):
    if ':' in policy:
        name, count = policy.split(':')
        restart_policy = {'Name': name, 'MaximumRetryCount': count}
    else:
        restart_policy = {'Name': policy, 'MaximumRetryCount': '0'}
    return restart_policy