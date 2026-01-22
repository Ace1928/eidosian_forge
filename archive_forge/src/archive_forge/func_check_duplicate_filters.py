import argparse
import collections
import getpass
import logging
import sys
from urllib import parse as urlparse
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import loading
from keystoneauth1 import session
from oslo_utils import importutils
import requests
import cinderclient
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import client
from cinderclient import exceptions as exc
from cinderclient import utils
def check_duplicate_filters(self, argv, filter):
    resource = RESOURCE_FILTERS[filter]
    filters = []
    for opt in range(len(argv)):
        if argv[opt].startswith('--'):
            if argv[opt] == '--filters':
                key, __ = argv[opt + 1].split('=')
                if key in resource:
                    filters.append(key)
            elif argv[opt][2:] in resource:
                filters.append(argv[opt][2:])
    if len(set(filters)) != len(filters):
        raise exc.CommandError('Filters are only allowed to be passed once.')