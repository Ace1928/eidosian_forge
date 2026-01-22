from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def update_cvo_tags(self, api_root, rest_api, headers, tag_name, tag_list):
    body = {}
    tags = []
    if tag_list is not None:
        for tag in tag_list:
            atag = {'tagKey': tag['label_key'] if tag_name == 'gcp_labels' else tag['tag_key'], 'tagValue': tag['label_value'] if tag_name == 'gcp_labels' else tag['tag_value']}
            tags.append(atag)
    body['tags'] = tags
    response, err, dummy = rest_api.put(api_root + 'user-tags', body, header=headers)
    if err is not None:
        return (False, 'Error: unexpected response on modifying tags: %s, %s' % (str(err), str(response)))
    return (True, None)