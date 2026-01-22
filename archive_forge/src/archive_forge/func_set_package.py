import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def set_package(self, package):
    """ set the name of the package in the last entry. """
    self._blocks[0].package = package