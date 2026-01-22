import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def register_rds_db_instance(self, stack_id, rds_db_instance_arn, db_user, db_password):
    """
        Registers an Amazon RDS instance with a stack.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type stack_id: string
        :param stack_id: The stack ID.

        :type rds_db_instance_arn: string
        :param rds_db_instance_arn: The Amazon RDS instance's ARN.

        :type db_user: string
        :param db_user: The database's master user name.

        :type db_password: string
        :param db_password: The database password.

        """
    params = {'StackId': stack_id, 'RdsDbInstanceArn': rds_db_instance_arn, 'DbUser': db_user, 'DbPassword': db_password}
    return self.make_request(action='RegisterRdsDbInstance', body=json.dumps(params))