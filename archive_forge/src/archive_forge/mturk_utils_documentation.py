import boto3
import os
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound

    Creates the actual HIT given the type and page to direct clients to.
    