from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def zone_range_exists(client, namespace, min, max, tag):
    """
    Returns true if a particular zone range exists
    Record format seems to be different than the docs state in 4.4.6
    { "_id" : ObjectId("60e2e7cff7c9d447440bb114"),
      "ns" : "records.users",
      "min" : { "zipcode" : "10001" },
      "max" : { "zipcode" : "10281" },
      "tag" : "NYC" }

    @client - MongoDB connection
    @namespace - In the form database.collection
    @min - The min range value
    @max - The max range value
    @tag - The tag or Zone name
    """
    query = {'ns': namespace, 'min': min, 'max': max, 'tag': tag}
    status = None
    result = client['config'].tags.find_one(query)
    if result:
        status = True
    else:
        status = False
    return status