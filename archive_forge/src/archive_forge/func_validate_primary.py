from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def validate_primary(self):
    isValid = True
    if self.primary_url is None:
        print(self.error_msg % (FAIL, PREFIX, 'url', 'primary', END))
        isValid = False
    if self.primary_user is None:
        print(self.error_msg % (FAIL, PREFIX, 'username', 'primary', END))
        isValid = False
    if self.primary_ca is None:
        print(self.error_msg % (FAIL, PREFIX, 'ca', 'primary', END))
        isValid = False
    return isValid