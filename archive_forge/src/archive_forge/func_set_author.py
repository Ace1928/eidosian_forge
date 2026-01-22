import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def set_author(self, author):
    """ set the author of the top changelog entry """
    self._blocks[0].author = author