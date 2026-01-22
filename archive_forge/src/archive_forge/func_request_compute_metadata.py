import enum
import os
import sys
import requests
from six.moves.urllib import request
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
def request_compute_metadata(path):
    """Returns GCE VM compute metadata."""
    gce_metadata_endpoint = 'http://' + os.environ.get(_GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')
    req = request.Request('%s/computeMetadata/v1/%s' % (gce_metadata_endpoint, path), headers={'Metadata-Flavor': 'Google'})
    info = request.urlopen(req).read()
    if isinstance(info, bytes):
        return info.decode('utf-8')
    else:
        return info