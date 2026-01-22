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
def get_v3_auth(self, v3_auth_url):
    username = self.options.os_username
    user_id = self.options.os_user_id
    user_domain_name = self.options.os_user_domain_name
    user_domain_id = self.options.os_user_domain_id
    password = self.options.os_password
    project_id = self.options.os_project_id
    project_name = self.options.os_project_name
    project_domain_name = self.options.os_project_domain_name
    project_domain_id = self.options.os_project_domain_id
    return v3_auth.Password(v3_auth_url, username=username, password=password, user_id=user_id, user_domain_name=user_domain_name, user_domain_id=user_domain_id, project_id=project_id, project_name=project_name, project_domain_name=project_domain_name, project_domain_id=project_domain_id)