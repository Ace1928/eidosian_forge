import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def delete_db_parameter_group(self, db_parameter_group_name):
    """
        Deletes a specified DBParameterGroup. The DBParameterGroup
        cannot be associated with any RDS instances to be deleted.
        The specified DB parameter group cannot be associated with any
        DB instances.

        :type db_parameter_group_name: string
        :param db_parameter_group_name:
        The name of the DB parameter group.

        Constraints:


        + Must be the name of an existing DB parameter group
        + You cannot delete a default DB parameter group
        + Cannot be associated with any DB instances

        """
    params = {'DBParameterGroupName': db_parameter_group_name}
    return self._make_request(action='DeleteDBParameterGroup', verb='POST', path='/', params=params)