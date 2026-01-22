from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_nss(self, rest_api, headers):
    """
        Get nss account
        """
    api = '/occm/api/accounts'
    response, error, dummy = rest_api.get(api, header=headers)
    if error is not None:
        return (None, 'Error: unexpected response on getting nss for cvo: %s, %s' % (str(error), str(response)))
    if len(response['nssAccounts']) == 0:
        return (None, 'Error: could not find any NSS account')
    return (response['nssAccounts'][0]['publicId'], None)