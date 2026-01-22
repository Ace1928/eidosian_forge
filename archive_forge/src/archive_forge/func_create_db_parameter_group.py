import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def create_db_parameter_group(self, db_parameter_group_name, db_parameter_group_family, description, tags=None):
    """
        Creates a new DB parameter group.

        A DB parameter group is initially created with the default
        parameters for the database engine used by the DB instance. To
        provide custom values for any of the parameters, you must
        modify the group after creating it using
        ModifyDBParameterGroup . Once you've created a DB parameter
        group, you need to associate it with your DB instance using
        ModifyDBInstance . When you associate a new DB parameter group
        with a running DB instance, you need to reboot the DB Instance
        for the new DB parameter group and associated settings to take
        effect.

        :type db_parameter_group_name: string
        :param db_parameter_group_name:
        The name of the DB parameter group.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens


        This value is stored as a lower-case string.

        :type db_parameter_group_family: string
        :param db_parameter_group_family: The DB parameter group family name. A
            DB parameter group can be associated with one and only one DB
            parameter group family, and can be applied only to a DB instance
            running a database engine and engine version compatible with that
            DB parameter group family.

        :type description: string
        :param description: The description for the DB parameter group.

        :type tags: list
        :param tags: A list of tags. Tags must be passed as tuples in the form
            [('key1', 'valueForKey1'), ('key2', 'valueForKey2')]

        """
    params = {'DBParameterGroupName': db_parameter_group_name, 'DBParameterGroupFamily': db_parameter_group_family, 'Description': description}
    if tags is not None:
        self.build_complex_list_params(params, tags, 'Tags.member', ('Key', 'Value'))
    return self._make_request(action='CreateDBParameterGroup', verb='POST', path='/', params=params)