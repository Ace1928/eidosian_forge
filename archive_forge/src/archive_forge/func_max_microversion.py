import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
@property
def max_microversion(self):
    """The maximum microversion supported by the endpoint.

        May be None.
        """
    return self.get('max_microversion')