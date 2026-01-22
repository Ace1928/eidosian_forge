import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def get_groups_from_server(cloud, server, server_vars):
    groups = []
    region = cloud.config.get_region_name('compute')
    cloud_name = cloud.name
    groups.append(cloud_name)
    groups.append(region)
    groups.append('%s_%s' % (cloud_name, region))
    group = server['metadata'].get('group')
    if group:
        groups.append(group)
    for extra_group in server['metadata'].get('groups', '').split(','):
        if extra_group:
            groups.append(extra_group)
    groups.append('instance-%s' % server['id'])
    for key in ('flavor', 'image'):
        if 'name' in server_vars[key]:
            groups.append('%s-%s' % (key, server_vars[key]['name']))
    for key, value in iter(server['metadata'].items()):
        groups.append('meta-%s_%s' % (key, value))
    az = server_vars.get('az', None)
    if az:
        groups.append(az)
        groups.append('%s_%s' % (region, az))
        groups.append('%s_%s_%s' % (cloud.name, region, az))
    return groups