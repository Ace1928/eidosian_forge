import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def subscribe_to_hits(hit_type_id, is_sandbox, sns_arn):
    client = get_mturk_client(is_sandbox)
    client.update_notification_settings(HITTypeId=hit_type_id, Notification={'Destination': sns_arn, 'Transport': 'SNS', 'Version': '2006-05-05', 'EventTypes': ['AssignmentAbandoned', 'AssignmentReturned', 'AssignmentSubmitted']}, Active=True)