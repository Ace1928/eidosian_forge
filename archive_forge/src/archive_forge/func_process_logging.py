from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
def process_logging(log_bucket, log_object_prefix):
    """Converts logging settings to S3 metadata dict."""
    clear_log_bucket = log_bucket == user_request_args_factory.CLEAR
    clear_log_object_prefix = log_object_prefix == user_request_args_factory.CLEAR
    if clear_log_bucket and clear_log_object_prefix:
        return user_request_args_factory.CLEAR
    logging_config = {}
    if log_bucket and (not clear_log_bucket):
        logging_config['TargetBucket'] = log_bucket
    if log_object_prefix and (not clear_log_object_prefix):
        logging_config['TargetPrefix'] = log_object_prefix
    return {'LoggingEnabled': logging_config}