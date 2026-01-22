from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time

    Returns the auth token used in kubectl
    This also sets the 'fetch' variable used in creating the kubectl
    