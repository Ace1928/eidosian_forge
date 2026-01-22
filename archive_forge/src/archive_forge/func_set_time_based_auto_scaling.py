import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def set_time_based_auto_scaling(self, instance_id, auto_scaling_schedule=None):
    """
        Specify the time-based auto scaling configuration for a
        specified instance. For more information, see `Managing Load
        with Time-based and Load-based Instances`_.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type instance_id: string
        :param instance_id: The instance ID.

        :type auto_scaling_schedule: dict
        :param auto_scaling_schedule: An `AutoScalingSchedule` with the
            instance schedule.

        """
    params = {'InstanceId': instance_id}
    if auto_scaling_schedule is not None:
        params['AutoScalingSchedule'] = auto_scaling_schedule
    return self.make_request(action='SetTimeBasedAutoScaling', body=json.dumps(params))