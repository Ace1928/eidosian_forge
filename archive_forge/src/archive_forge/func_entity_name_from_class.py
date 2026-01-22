from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
@property
def entity_name_from_class(self):
    """
        The entity name derived from the class name.

        The class name must follow the following name convention:

        * It starts with ``Foreman`` or ``Katello``.
        * It ends with ``Module``.

        This will convert the class name ``ForemanMyEntityModule`` to the entity name ``my_entity``.

        Examples:

        * ``ForemanArchitectureModule`` => ``architecture``
        * ``ForemanProvisioningTemplateModule`` => ``provisioning_template``
        * ``KatelloProductMudule`` => ``product``
        """
    class_name = re.sub('(?<=[a-z])[A-Z]|[A-Z](?=[^A-Z])', '_\\g<0>', self.__class__.__name__).lower().strip('_')
    return '_'.join(class_name.split('_')[1:-1])