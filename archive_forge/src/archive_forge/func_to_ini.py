from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.six.moves import StringIO
from ansible.module_utils.six.moves.configparser import ConfigParser
from ansible.module_utils.common.text.converters import to_native
def to_ini(obj):
    """ Read the given dict and return an INI formatted string """
    if not isinstance(obj, Mapping):
        raise AnsibleFilterError(f'to_ini requires a dict, got {type(obj)}')
    ini_parser = IniParser()
    try:
        ini_parser.read_dict(obj)
    except Exception as ex:
        raise AnsibleFilterError(f'to_ini failed to parse given dict:{to_native(ex)}', orig_exc=ex)
    if obj == dict():
        raise AnsibleFilterError('to_ini received an empty dict. An empty dict cannot be converted.')
    config = StringIO()
    ini_parser.write(config)
    return ''.join(config.getvalue().rsplit(config.getvalue()[-1], 1))