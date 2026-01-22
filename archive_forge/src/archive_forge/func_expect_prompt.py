from __future__ import (absolute_import, division, print_function)
import shlex
from abc import abstractmethod
from random import choice
from string import ascii_lowercase
from gettext import dgettext
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins import AnsiblePlugin
def expect_prompt(self):
    """This function assists connection plugins in determining if they need to wait for
        a prompt. Both a prompt and a password are required.
        """
    return self.prompt and self.get_option('become_pass')