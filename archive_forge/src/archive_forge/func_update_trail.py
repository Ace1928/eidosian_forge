import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudtrail import exceptions
from boto.compat import json
def update_trail(self, name, s3_bucket_name=None, s3_key_prefix=None, sns_topic_name=None, include_global_service_events=None, cloud_watch_logs_log_group_arn=None, cloud_watch_logs_role_arn=None):
    """
        From the command line, use `update-subscription`.

        Updates the settings that specify delivery of log files.
        Changes to a trail do not require stopping the CloudTrail
        service. Use this action to designate an existing bucket for
        log delivery. If the existing bucket has previously been a
        target for CloudTrail log files, an IAM policy exists for the
        bucket.

        :type name: string
        :param name: Specifies the name of the trail.

        :type s3_bucket_name: string
        :param s3_bucket_name: Specifies the name of the Amazon S3 bucket
            designated for publishing log files.

        :type s3_key_prefix: string
        :param s3_key_prefix: Specifies the Amazon S3 key prefix that precedes
            the name of the bucket you have designated for log file delivery.

        :type sns_topic_name: string
        :param sns_topic_name: Specifies the name of the Amazon SNS topic
            defined for notification of log file delivery.

        :type include_global_service_events: boolean
        :param include_global_service_events: Specifies whether the trail is
            publishing events from global services such as IAM to the log
            files.

        :type cloud_watch_logs_log_group_arn: string
        :param cloud_watch_logs_log_group_arn: Specifies a log group name using
            an Amazon Resource Name (ARN), a unique identifier that represents
            the log group to which CloudTrail logs will be delivered. Not
            required unless you specify CloudWatchLogsRoleArn.

        :type cloud_watch_logs_role_arn: string
        :param cloud_watch_logs_role_arn: Specifies the role for the CloudWatch
            Logs endpoint to assume to write to a users log group.

        """
    params = {'Name': name}
    if s3_bucket_name is not None:
        params['S3BucketName'] = s3_bucket_name
    if s3_key_prefix is not None:
        params['S3KeyPrefix'] = s3_key_prefix
    if sns_topic_name is not None:
        params['SnsTopicName'] = sns_topic_name
    if include_global_service_events is not None:
        params['IncludeGlobalServiceEvents'] = include_global_service_events
    if cloud_watch_logs_log_group_arn is not None:
        params['CloudWatchLogsLogGroupArn'] = cloud_watch_logs_log_group_arn
    if cloud_watch_logs_role_arn is not None:
        params['CloudWatchLogsRoleArn'] = cloud_watch_logs_role_arn
    return self.make_request(action='UpdateTrail', body=json.dumps(params))