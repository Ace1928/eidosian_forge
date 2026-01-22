import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def create_node_group_templates(client, app, parsed_args, flavor_id, configs, shares):
    if app.api_version['data_processing'] == '2':
        data = client.node_group_templates.create(name=parsed_args.name, plugin_name=parsed_args.plugin, plugin_version=parsed_args.plugin_version, flavor_id=flavor_id, description=parsed_args.description, volumes_per_node=parsed_args.volumes_per_node, volumes_size=parsed_args.volumes_size, node_processes=parsed_args.processes, floating_ip_pool=parsed_args.floating_ip_pool, security_groups=parsed_args.security_groups, auto_security_group=parsed_args.auto_security_group, availability_zone=parsed_args.availability_zone, volume_type=parsed_args.volumes_type, is_proxy_gateway=parsed_args.proxy_gateway, volume_local_to_instance=parsed_args.volumes_locality, use_autoconfig=parsed_args.autoconfig, is_public=parsed_args.public, is_protected=parsed_args.protected, node_configs=configs, shares=shares, volumes_availability_zone=parsed_args.volumes_availability_zone, volume_mount_prefix=parsed_args.volumes_mount_prefix, boot_from_volume=parsed_args.boot_from_volume, boot_volume_type=parsed_args.boot_volume_type, boot_volume_availability_zone=parsed_args.boot_volume_availability_zone, boot_volume_local_to_instance=parsed_args.boot_volume_local_to_instance).to_dict()
    else:
        data = client.node_group_templates.create(name=parsed_args.name, plugin_name=parsed_args.plugin, hadoop_version=parsed_args.plugin_version, flavor_id=flavor_id, description=parsed_args.description, volumes_per_node=parsed_args.volumes_per_node, volumes_size=parsed_args.volumes_size, node_processes=parsed_args.processes, floating_ip_pool=parsed_args.floating_ip_pool, security_groups=parsed_args.security_groups, auto_security_group=parsed_args.auto_security_group, availability_zone=parsed_args.availability_zone, volume_type=parsed_args.volumes_type, is_proxy_gateway=parsed_args.proxy_gateway, volume_local_to_instance=parsed_args.volumes_locality, use_autoconfig=parsed_args.autoconfig, is_public=parsed_args.public, is_protected=parsed_args.protected, node_configs=configs, shares=shares, volumes_availability_zone=parsed_args.volumes_availability_zone, volume_mount_prefix=parsed_args.volumes_mount_prefix).to_dict()
    return data