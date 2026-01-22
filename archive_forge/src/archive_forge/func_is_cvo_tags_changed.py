from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def is_cvo_tags_changed(self, rest_api, headers, parameters, tag_name):
    """
        Since tags/laabels are CVO optional parameters, this function needs to cover with/without tags/labels on both lists
        """
    current, error = self.get_working_environment_details(rest_api, headers)
    if error is not None:
        return (None, 'Error:  Cannot find working environment %s error: %s' % (self.parameters['working_environment_id'], str(error)))
    self.set_api_root_path(current, rest_api)
    if 'userTags' not in current or len(current['userTags']) == 0:
        return (tag_name in parameters, None)
    if tag_name == 'gcp_labels':
        if tag_name in parameters:
            return self.compare_gcp_labels(current['userTags'], parameters[tag_name], current['isHA'])
        tag_keys = list(current['userTags'].keys())
        user_tag_keys = [key for key in tag_keys if key not in ('count-down', 'gcp_resource_id', 'partner-platform-serial-number')]
        if not user_tag_keys:
            return (False, None)
        else:
            return (None, 'Error:  Cannot remove current gcp_labels')
    if tag_name not in parameters:
        return (True, None)
    else:
        return self.compare_cvo_tags_labels(current['userTags'], parameters[tag_name])