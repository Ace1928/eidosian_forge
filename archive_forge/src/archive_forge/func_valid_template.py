from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def valid_template(port, template):
    """ Test if the user provided Jinja template is valid.

    :param port: User specified port.
    :param template: Contents of Jinja template.
    :return: true or False
    """
    valid = True
    regex = '^interface Ethernet%s' % port
    match = re.match(regex, template, re.M)
    if not match:
        valid = False
    return valid