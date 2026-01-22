from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def semanage_port_get_ports(seport, setype, proto, local):
    """ Get the list of ports that have the specified type definition.

    :param community.general.seport: Instance of seobject.portRecords

    :type setype: str
    :param setype: SELinux type.

    :type proto: str
    :param proto: Protocol ('tcp' or 'udp')

    :rtype: list
    :return: List of ports that have the specified SELinux type.
    """
    records = seport.get_all_by_type(locallist=local)
    if (setype, proto) in records:
        return records[setype, proto]
    else:
        return []