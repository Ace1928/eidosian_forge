from __future__ import absolute_import, division, print_function
import json
import logging
import re
from pprint import pformat
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
 Checks to see if a passed in port arg is present on a different host 