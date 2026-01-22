import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_server_group(id, name, policies):
    return json.loads(json.dumps({'id': id, 'name': name, 'policies': policies, 'members': [], 'metadata': {}}))