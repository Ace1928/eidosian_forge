import copy
import urllib
from oslo_serialization import jsonutils
from keystoneauth1 import exceptions
from mistralclient import utils
def get_contents_if_file(self, contents_or_file_name):
    if self.enforce_raw_definitions:
        return contents_or_file_name
    else:
        return utils.get_contents_if_file(contents_or_file_name)