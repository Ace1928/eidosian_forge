from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import urlparse
@staticmethod
def parse_rommon_file_location(rommon_file_location):
    rommon_url = urlparse(rommon_file_location)
    if rommon_url.scheme != 'tftp':
        raise ValueError('The ROMMON image must be downloaded from TFTP server, other protocols are not supported.')
    return (rommon_url.netloc, rommon_url.path)