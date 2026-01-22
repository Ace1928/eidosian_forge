from __future__ import absolute_import, division, print_function
import traceback
from os import getenv
from os.path import isfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native

    Method that creates/deletes issues depending whether they exist and the state desired

    The credentials should be passed via environment variables:
        - TAIGA_TOKEN
        - TAIGA_USERNAME and TAIGA_PASSWORD

    Returns a tuple with these elements:
        - A boolean representing the success of the operation
        - A descriptive message
        - A dict with the issue attributes, in case of issue creation, otherwise empty dict
    