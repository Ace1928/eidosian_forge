import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_flavor(flavor_id, name, ram=100, disk=1600, vcpus=24):
    return {u'OS-FLV-DISABLED:disabled': False, u'OS-FLV-EXT-DATA:ephemeral': 0, u'disk': disk, u'id': flavor_id, u'links': [{u'href': u'{endpoint}/flavors/{id}'.format(endpoint=COMPUTE_ENDPOINT, id=flavor_id), u'rel': u'self'}, {u'href': u'{endpoint}/flavors/{id}'.format(endpoint=COMPUTE_ENDPOINT, id=flavor_id), u'rel': u'bookmark'}], u'name': name, u'os-flavor-access:is_public': True, u'ram': ram, u'rxtx_factor': 1.0, u'swap': 0, u'vcpus': vcpus}