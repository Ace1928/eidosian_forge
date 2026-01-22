from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def is_valid_password(self, password):
    """
        Check if a password is valid.
        Args:
            self (object): An instance of a class that provides access to Cisco Catalyst Center.
            password (str): The password to be validated.
        Returns:
            bool: True if the password is valid, False otherwise.
        Description:
            The function checks the validity of a password based on the following criteria:
            - Minimum 8 characters.
            - At least one lowercase letter.
            - At least one uppercase letter.
            - At least one digit.
            - At least one special character
        """
    pattern = '^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[-=\\\\;,./~!@#$%^&*()_+{}[\\]|:?]).{8,}$'
    return re.match(pattern, password) is not None