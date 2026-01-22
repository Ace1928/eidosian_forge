import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def remove_tags_from_resource(self, resource_name, tag_keys):
    """
        Removes metadata tags from an Amazon RDS resource.

        For an overview on tagging an Amazon RDS resource, see
        `Tagging Amazon RDS Resources`_.

        :type resource_name: string
        :param resource_name: The Amazon RDS resource the tags will be removed
            from. This value is an Amazon Resource Name (ARN). For information
            about creating an ARN, see ` Constructing an RDS Amazon Resource
            Name (ARN)`_.

        :type tag_keys: list
        :param tag_keys: The tag key (name) of the tag to be removed.

        """
    params = {'ResourceName': resource_name}
    self.build_list_params(params, tag_keys, 'TagKeys.member')
    return self._make_request(action='RemoveTagsFromResource', verb='POST', path='/', params=params)