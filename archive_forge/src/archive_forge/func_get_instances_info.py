import argparse
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
def get_instances_info(instances):
    instances_info = []
    for instance in instances:
        instance_info = instance.to_dict()
        instance_info['flavor_id'] = instance.flavor['id']
        instance_info['size'] = '-'
        if 'volume' in instance_info:
            instance_info['size'] = instance_info['volume']['size']
        instance_info['role'] = ''
        if 'replica_of' in instance_info:
            instance_info['role'] = 'replica'
        if 'replicas' in instance_info:
            instance_info['role'] = 'primary'
        if 'datastore' in instance_info:
            if instance.datastore.get('version'):
                instance_info['datastore_version'] = instance.datastore['version']
            instance_info['datastore'] = instance.datastore['type']
        if 'access' in instance_info:
            instance_info['public'] = instance_info['access'].get('is_public', False)
        if 'addresses' not in instance_info:
            instance_info['addresses'] = ''
        if 'operating_status' not in instance_info:
            instance_info['operating_status'] = ''
        instances_info.append(instance_info)
    return instances_info