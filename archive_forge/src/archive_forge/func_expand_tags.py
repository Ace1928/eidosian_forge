from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_tags(eg_launchspec, tags):
    if tags is not None:
        eg_tags = []
        for tag in tags:
            eg_tag = spotinst.aws_elastigroup.Tag()
            if tag:
                eg_tag.tag_key, eg_tag.tag_value = list(tag.items())[0]
            eg_tags.append(eg_tag)
        if len(eg_tags) > 0:
            eg_launchspec.tags = eg_tags