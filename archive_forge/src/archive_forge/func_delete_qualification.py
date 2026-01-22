import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def delete_qualification(qualification_id, is_sandbox):
    """
    Deletes a qualification by id.
    """
    client = get_mturk_client(is_sandbox)
    client.delete_qualification_type(QualificationTypeId=qualification_id)