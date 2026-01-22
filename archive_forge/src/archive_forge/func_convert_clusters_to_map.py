from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def convert_clusters_to_map(clusters):
    cmap = {}
    for cluster in clusters:
        cmap[cluster['name']] = cluster
        del cmap[cluster['name']]['name']
    return cmap