import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def set_permission(self, stack_id, iam_user_arn, allow_ssh=None, allow_sudo=None, level=None):
    """
        Specifies a user's permissions. For more information, see
        `Security and Permissions`_.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type stack_id: string
        :param stack_id: The stack ID.

        :type iam_user_arn: string
        :param iam_user_arn: The user's IAM ARN.

        :type allow_ssh: boolean
        :param allow_ssh: The user is allowed to use SSH to communicate with
            the instance.

        :type allow_sudo: boolean
        :param allow_sudo: The user is allowed to use **sudo** to elevate
            privileges.

        :type level: string
        :param level: The user's permission level, which must be set to one of
            the following strings. You cannot set your own permissions level.

        + `deny`
        + `show`
        + `deploy`
        + `manage`
        + `iam_only`


        For more information on the permissions associated with these levels,
            see `Managing User Permissions`_

        """
    params = {'StackId': stack_id, 'IamUserArn': iam_user_arn}
    if allow_ssh is not None:
        params['AllowSsh'] = allow_ssh
    if allow_sudo is not None:
        params['AllowSudo'] = allow_sudo
    if level is not None:
        params['Level'] = level
    return self.make_request(action='SetPermission', body=json.dumps(params))