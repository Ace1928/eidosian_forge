import datetime
import getpass
import json
import logging
import os
import subprocess
import threading
import time
from collections import namedtuple
from copy import deepcopy
from hashlib import sha1
from dateutil.parser import parse
from dateutil.tz import tzlocal, tzutc
import botocore.compat
import botocore.configloader
from botocore import UNSIGNED
from botocore.compat import compat_shell_split, total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.tokens import SSOTokenProvider
from botocore.utils import (
def fetch_creds():
    try:
        response = self._fetcher.retrieve_full_uri(full_uri, headers=headers)
    except MetadataRetrievalError as e:
        logger.debug('Error retrieving container metadata: %s', e, exc_info=True)
        raise CredentialRetrievalError(provider=self.METHOD, error_msg=str(e))
    return {'access_key': response['AccessKeyId'], 'secret_key': response['SecretAccessKey'], 'token': response['Token'], 'expiry_time': response['Expiration']}