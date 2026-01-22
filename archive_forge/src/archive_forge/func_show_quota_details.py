import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
@debtcollector.renames.renamed_kwarg('tenant_id', 'project_id', replace=True)
def show_quota_details(self, project_id, **_params):
    """Fetch information of a certain project's quota details."""
    return self.get(self.quota_details_path % project_id, params=_params)