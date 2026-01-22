from botocore.exceptions import ClientError
from boto3.s3.transfer import create_transfer_manager
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.s3.transfer import ProgressCallbackInvoker
from boto3 import utils
def inject_object_summary_methods(class_attributes, **kwargs):
    utils.inject_attribute(class_attributes, 'load', object_summary_load)