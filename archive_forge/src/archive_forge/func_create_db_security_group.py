import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def create_db_security_group(self, db_security_group_name, db_security_group_description, tags=None):
    """
        Creates a new DB security group. DB security groups control
        access to a DB instance.

        :type db_security_group_name: string
        :param db_security_group_name: The name for the DB security group. This
            value is stored as a lowercase string.
        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens
        + Must not be "Default"
        + May not contain spaces


        Example: `mysecuritygroup`

        :type db_security_group_description: string
        :param db_security_group_description: The description for the DB
            security group.

        :type tags: list
        :param tags: A list of tags. Tags must be passed as tuples in the form
            [('key1', 'valueForKey1'), ('key2', 'valueForKey2')]

        """
    params = {'DBSecurityGroupName': db_security_group_name, 'DBSecurityGroupDescription': db_security_group_description}
    if tags is not None:
        self.build_complex_list_params(params, tags, 'Tags.member', ('Key', 'Value'))
    return self._make_request(action='CreateDBSecurityGroup', verb='POST', path='/', params=params)