import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound
def find_or_create_qualification(qualification_name, description, is_sandbox, must_be_owned=True):
    """
    Query amazon to find the existing qualification name, return the Id.

    If it exists and must_be_owned is true but we don't own it, this prints an error and
    returns none. If it doesn't exist, the qualification is created
    """
    qual_id = find_qualification(qualification_name, is_sandbox, must_be_owned=must_be_owned)
    if qual_id is False:
        return None
    if qual_id is not None:
        return qual_id
    client = get_mturk_client(is_sandbox)
    response = client.create_qualification_type(Name=qualification_name, Description=description, QualificationTypeStatus='Active')
    return response['QualificationType']['QualificationTypeId']