from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import urlparse
def install_ftd_image(self, params):
    line = self._ftd.ssh_console(ip=params['console_ip'], port=params['console_port'], username=params['console_username'], password=params['console_password'])
    try:
        rommon_server, rommon_path = self.parse_rommon_file_location(params['rommon_file_location'])
        line.rommon_to_new_image(rommon_tftp_server=rommon_server, rommon_image=rommon_path, pkg_image=params['image_file_location'], uut_ip=params['device_ip'], uut_netmask=params['device_netmask'], uut_gateway=params['device_gateway'], dns_server=params['dns_server'], search_domains=params['search_domains'], hostname=params['device_hostname'])
    finally:
        line.disconnect()