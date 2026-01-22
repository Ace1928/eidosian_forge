from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def compare_cvo_tags_labels(self, current_tags, user_tags):
    """
        Compare exiting tags/labels and user input tags/labels to see if there is a change
        gcp_labels: label_key, label_value
        aws_tag/azure_tag: tag_key, tag_label
        """
    tag_keys = list(current_tags.keys())
    user_tag_keys = [key for key in tag_keys if key != 'DeployedByOccm']
    current_len = len(user_tag_keys)
    resp, error = self.user_tag_key_unique(user_tags, 'tag_key')
    if error is not None:
        return (None, error)
    if len(user_tags) != current_len:
        return (True, None)
    for item in user_tags:
        if item['tag_key'] in current_tags and item['tag_value'] != current_tags[item['tag_key']]:
            return (True, None)
        elif item['tag_key'] not in current_tags:
            return (True, None)
    return (False, None)