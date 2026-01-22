from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def get_multiple_pollers_results(self, pollers, wait=0.05):
    """
        Consistent method of waiting on and retrieving results from multiple Azure's long poller

        :param pollers list of Azure poller object
        :param wait Period of time to wait for the long running operation to complete.
        :return list of object resulting from the original request
        """

    def _continue_polling():
        return not all((poller.done() for poller in pollers))
    try:
        while _continue_polling():
            for poller in pollers:
                if poller.done():
                    continue
                self.log('Waiting for {0} sec'.format(wait))
                poller.wait(timeout=wait)
        return [poller.result() for poller in pollers]
    except Exception as exc:
        self.log(str(exc))
        raise