import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def delete_db_security_group(self, db_security_group_name):
    """
        Deletes a DB security group.
        The specified DB security group must not be associated with
        any DB instances.

        :type db_security_group_name: string
        :param db_security_group_name:
        The name of the DB security group to delete.

        You cannot delete the default DB security group.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens
        + Must not be "Default"
        + May not contain spaces

        """
    params = {'DBSecurityGroupName': db_security_group_name}
    return self._make_request(action='DeleteDBSecurityGroup', verb='POST', path='/', params=params)