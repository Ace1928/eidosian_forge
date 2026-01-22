from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.storage.hpe3par import hpe3par
def create_cpg(client_obj, cpg_name, domain, growth_increment, growth_limit, growth_warning, raid_type, set_size, high_availability, disk_type):
    try:
        if not validate_set_size(raid_type, set_size):
            return (False, False, 'Set size %s not part of RAID set %s' % (set_size, raid_type))
        if not client_obj.cpgExists(cpg_name):
            disk_patterns = []
            if disk_type:
                disk_type = getattr(client.HPE3ParClient, disk_type)
                disk_patterns = [{'diskType': disk_type}]
            ld_layout = {'RAIDType': raid_type, 'setSize': set_size, 'HA': high_availability, 'diskPatterns': disk_patterns}
            ld_layout = cpg_ldlayout_map(ld_layout)
            if growth_increment is not None:
                growth_increment = hpe3par.convert_to_binary_multiple(growth_increment)
            if growth_limit is not None:
                growth_limit = hpe3par.convert_to_binary_multiple(growth_limit)
            if growth_warning is not None:
                growth_warning = hpe3par.convert_to_binary_multiple(growth_warning)
            optional = {'domain': domain, 'growthIncrementMiB': growth_increment, 'growthLimitMiB': growth_limit, 'usedLDWarningAlertMiB': growth_warning, 'LDLayout': ld_layout}
            client_obj.createCPG(cpg_name, optional)
        else:
            return (True, False, 'CPG already present')
    except exceptions.ClientException as e:
        return (False, False, 'CPG creation failed | %s' % e)
    return (True, True, 'Created CPG %s successfully.' % cpg_name)