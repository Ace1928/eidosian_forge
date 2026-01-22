import copy
from collections import deque
from pprint import pformat
from botocore.awsrequest import AWSResponse
from botocore.exceptions import (
from botocore.validate import validate_parameters
def assert_no_pending_responses(self):
    """
        Asserts that all expected calls were made.
        """
    remaining = len(self._queue)
    if remaining != 0:
        raise AssertionError(f'{remaining} responses remaining in queue.')