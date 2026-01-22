import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
def set_management_url(self, url):
    self.client.management_url = url