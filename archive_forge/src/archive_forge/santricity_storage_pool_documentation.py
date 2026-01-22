from __future__ import (absolute_import, division, print_function)
import re
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleError
from itertools import product
Generate list of strings determined by a pattern