from __future__ import (absolute_import, division, print_function)
import configparser
import os
import re
from io import StringIO
from collections import defaultdict
from collections.abc import MutableSequence
from ansible.errors import AnsibleLookupError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.plugins.lookup import LookupBase
Safely split parameter term to preserve spaces