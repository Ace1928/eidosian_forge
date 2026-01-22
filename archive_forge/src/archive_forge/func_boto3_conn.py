import json
import os
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import text_type
from .common import get_collection_info
from .exceptions import AnsibleBotocoreError
from .retries import AWSRetry
def boto3_conn(module, conn_type=None, resource=None, region=None, endpoint=None, **params):
    """
    Builds a boto3 resource/client connection cleanly wrapping the most common failures.
    Handles:
        ValueError,
        botocore.exceptions.ProfileNotFound, botocore.exceptions.PartialCredentialsError,
        botocore.exceptions.NoCredentialsError, botocore.exceptions.ConfigParseError,
        botocore.exceptions.NoRegionError
    """
    try:
        return _boto3_conn(conn_type=conn_type, resource=resource, region=region, endpoint=endpoint, **params)
    except ValueError as e:
        module.fail_json(msg=f"Couldn't connect to AWS: {to_native(e)}")
    except (botocore.exceptions.ProfileNotFound, botocore.exceptions.PartialCredentialsError, botocore.exceptions.NoCredentialsError, botocore.exceptions.ConfigParseError) as e:
        module.fail_json(msg=to_native(e))
    except botocore.exceptions.NoRegionError:
        module.fail_json(msg=f'The {module._name} module requires a region and none was found in configuration, environment variables or module parameters')