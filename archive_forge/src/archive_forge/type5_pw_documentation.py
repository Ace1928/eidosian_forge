from __future__ import absolute_import, division, print_function
import string
from ansible.errors import AnsibleFilterError
from ansible.module_utils._text import to_text
from ansible.module_utils.six import string_types
from ansible.utils.encrypt import passlib_or_crypt, random_password

The type5_pw plugin code
