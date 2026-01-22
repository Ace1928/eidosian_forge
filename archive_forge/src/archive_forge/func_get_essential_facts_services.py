from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible_collections.community.docker.plugins.module_utils.common import RequestException
from ansible_collections.community.docker.plugins.module_utils.util import (
@staticmethod
def get_essential_facts_services(item):
    object_essentials = dict()
    object_essentials['ID'] = item['ID']
    object_essentials['Name'] = item['Spec']['Name']
    if 'Replicated' in item['Spec']['Mode']:
        object_essentials['Mode'] = 'Replicated'
        object_essentials['Replicas'] = item['Spec']['Mode']['Replicated']['Replicas']
    elif 'Global' in item['Spec']['Mode']:
        object_essentials['Mode'] = 'Global'
        object_essentials['Replicas'] = None
    object_essentials['Image'] = item['Spec']['TaskTemplate']['ContainerSpec']['Image']
    if item['Spec'].get('EndpointSpec') and 'Ports' in item['Spec']['EndpointSpec']:
        object_essentials['Ports'] = item['Spec']['EndpointSpec']['Ports']
    else:
        object_essentials['Ports'] = []
    return object_essentials