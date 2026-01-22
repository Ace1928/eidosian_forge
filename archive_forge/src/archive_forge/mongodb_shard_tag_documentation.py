from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (

    Remove a zone tag from a shard.
    @client - MongoDB connection
    @shard - The shard name
    @tag - The tag or Zone name
    