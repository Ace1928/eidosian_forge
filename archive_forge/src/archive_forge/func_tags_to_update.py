from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def tags_to_update(self, tags, action):
    """ Create a list of tags we need to update in ManageIQ.

        Returns:
            Whether or not a change took place and a message describing the
            operation executed.
        """
    tags_to_post = []
    assigned_tags = self.query_resource_tags()
    assigned_tags_set = set([tag['full_name'] for tag in assigned_tags])
    for tag in tags:
        assigned = self.full_tag_name(tag) in assigned_tags_set
        if assigned and action == 'unassign':
            tags_to_post.append(tag)
        elif not assigned and action == 'assign':
            tags_to_post.append(tag)
    return tags_to_post