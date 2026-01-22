from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
@property
def twotuples(self):
    error = [('error', self.error)]
    if self.description:
        error.append(('error_description', self.description))
    if self.uri:
        error.append(('error_uri', self.uri))
    if self.state:
        error.append(('state', self.state))
    return error