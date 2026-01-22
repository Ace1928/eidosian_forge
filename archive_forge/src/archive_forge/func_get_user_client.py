import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
@classmethod
def get_user_client(cls):
    user_client = base.CLIClient(username=CONF.username, password=CONF.password, tenant_name=CONF.tenant_name, uri=CONF.auth_url, cli_dir=CONF.manila_exec_dir, insecure=CONF.insecure, project_domain_name=CONF.project_domain_name or None, project_domain_id=CONF.project_domain_id or None, user_domain_name=CONF.user_domain_name or None, user_domain_id=CONF.user_domain_id or None)
    return user_client