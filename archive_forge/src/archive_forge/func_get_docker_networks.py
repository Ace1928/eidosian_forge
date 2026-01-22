from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def get_docker_networks(networks, network_ids):
    """
    Validate a list of network names or a list of network dictionaries.
    Network names will be resolved to ids by using the network_ids mapping.
    """
    if networks is None:
        return None
    parsed_networks = []
    for network in networks:
        if isinstance(network, string_types):
            parsed_network = {'name': network}
        elif isinstance(network, dict):
            if 'name' not in network:
                raise TypeError('"name" is required when networks are passed as dictionaries.')
            name = network.pop('name')
            parsed_network = {'name': name}
            aliases = network.pop('aliases', None)
            if aliases is not None:
                if not isinstance(aliases, list):
                    raise TypeError('"aliases" network option is only allowed as a list')
                if not all((isinstance(alias, string_types) for alias in aliases)):
                    raise TypeError('Only strings are allowed as network aliases.')
                parsed_network['aliases'] = aliases
            options = network.pop('options', None)
            if options is not None:
                if not isinstance(options, dict):
                    raise TypeError('Only dict is allowed as network options.')
                parsed_network['options'] = clean_dict_booleans_for_docker_api(options)
            if network:
                invalid_keys = ', '.join(network.keys())
                raise TypeError('%s are not valid keys for the networks option' % invalid_keys)
        else:
            raise TypeError('Only a list of strings or dictionaries are allowed to be passed as networks.')
        network_name = parsed_network.pop('name')
        try:
            parsed_network['id'] = network_ids[network_name]
        except KeyError as e:
            raise ValueError('Could not find a network named: %s.' % e)
        parsed_networks.append(parsed_network)
    return parsed_networks or []