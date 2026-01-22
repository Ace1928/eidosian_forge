import os
import re
import warnings
import boto
from boto.compat import expanduser, ConfigParser, NoOptionError, NoSectionError, StringIO
def save_system_option(self, section, option, value):
    self.save_option(BotoConfigPath, section, option, value)