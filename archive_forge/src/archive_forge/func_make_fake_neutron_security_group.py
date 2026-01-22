import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_neutron_security_group(id, name, description, rules, stateful=True, project_id=None):
    if not rules:
        rules = []
    if not project_id:
        project_id = PROJECT_ID
    return json.loads(json.dumps({'id': id, 'name': name, 'description': description, 'stateful': stateful, 'project_id': project_id, 'tenant_id': project_id, 'security_group_rules': rules}))