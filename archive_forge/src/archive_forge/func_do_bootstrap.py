import argparse
import datetime
import os
import sys
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_log import log
from oslo_serialization import jsonutils
import pbr.version
from keystone.cmd import bootstrap
from keystone.cmd import doctor
from keystone.cmd import idutils
from keystone.common import driver_hints
from keystone.common import fernet_utils
from keystone.common import jwt_utils
from keystone.common import sql
from keystone.common.sql import upgrades
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.federation import idp
from keystone.federation import utils as mapping_engine
from keystone.i18n import _
from keystone.server import backends
def do_bootstrap(self):
    """Perform the bootstrap actions.

        Create bootstrap user, project, and role so that CMS, humans, or
        scripts can continue to perform initial setup (domains, projects,
        services, endpoints, etc) of Keystone when standing up a new
        deployment.
        """
    self.username = os.environ.get('OS_BOOTSTRAP_USERNAME') or CONF.command.bootstrap_username
    self.project_name = os.environ.get('OS_BOOTSTRAP_PROJECT_NAME') or CONF.command.bootstrap_project_name
    self.role_name = os.environ.get('OS_BOOTSTRAP_ROLE_NAME') or CONF.command.bootstrap_role_name
    self.password = os.environ.get('OS_BOOTSTRAP_PASSWORD') or CONF.command.bootstrap_password
    self.service_name = os.environ.get('OS_BOOTSTRAP_SERVICE_NAME') or CONF.command.bootstrap_service_name
    self.admin_url = os.environ.get('OS_BOOTSTRAP_ADMIN_URL') or CONF.command.bootstrap_admin_url
    self.public_url = os.environ.get('OS_BOOTSTRAP_PUBLIC_URL') or CONF.command.bootstrap_public_url
    self.internal_url = os.environ.get('OS_BOOTSTRAP_INTERNAL_URL') or CONF.command.bootstrap_internal_url
    self.region_id = os.environ.get('OS_BOOTSTRAP_REGION_ID') or CONF.command.bootstrap_region_id
    self.service_id = None
    self.endpoints = None
    if self.password is None:
        print(_('ERROR: Either --bootstrap-password argument or OS_BOOTSTRAP_PASSWORD must be set.'))
        sys.exit(1)
    self.bootstrapper.admin_password = self.password
    self.bootstrapper.admin_username = self.username
    self.bootstrapper.project_name = self.project_name
    self.bootstrapper.admin_role_name = self.role_name
    self.bootstrapper.service_name = self.service_name
    self.bootstrapper.service_id = self.service_id
    self.bootstrapper.admin_url = self.admin_url
    self.bootstrapper.public_url = self.public_url
    self.bootstrapper.internal_url = self.internal_url
    self.bootstrapper.region_id = self.region_id
    if CONF.command.no_immutable_roles:
        self.bootstrapper.immutable_roles = False
    else:
        self.bootstrapper.immutable_roles = True
    self.bootstrapper.bootstrap()
    self.service_role_id = self.bootstrapper.service_role_id
    self.reader_role_id = self.bootstrapper.reader_role_id
    self.member_role_id = self.bootstrapper.member_role_id
    self.manager_role_id = self.bootstrapper.manager_role_id
    self.role_id = self.bootstrapper.admin_role_id
    self.project_id = self.bootstrapper.project_id