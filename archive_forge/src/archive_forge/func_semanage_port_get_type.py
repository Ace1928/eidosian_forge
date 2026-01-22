from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def semanage_port_get_type(seport, port, proto):
    """ Get the SELinux type of the specified port.

    :param community.general.seport: Instance of seobject.portRecords

    :type port: str
    :param port: Port or port range (example: "8080", "8080-9090")

    :type proto: str
    :param proto: Protocol ('tcp' or 'udp')

    :rtype: tuple
    :return: Tuple containing the SELinux type and MLS/MCS level, or None if not found.
    """
    if isinstance(port, str):
        ports = port.split('-', 1)
        if len(ports) == 1:
            ports.extend(ports)
    else:
        ports = (port, port)
    key = (int(ports[0]), int(ports[1]), proto)
    records = seport.get_all()
    return records.get(key)