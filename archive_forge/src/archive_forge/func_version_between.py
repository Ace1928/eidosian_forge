import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def version_between(min_version, max_version, candidate):
    """Determine whether a candidate version is within a specified range.

    :param min_version: The minimum version that is acceptable.
                        None/empty indicates no lower bound.
    :param max_version: The maximum version that is acceptable.
                        None/empty indicates no upper bound.
    :param candidate: Candidate version to test.  May not be None/empty.
    :return: True if candidate is between min_version and max_version; False
             otherwise.
    :raises ValueError: If candidate is None.
    :raises TypeError: If any input cannot be normalized.
    """
    if not candidate:
        raise ValueError('candidate is required.')
    candidate = normalize_version_number(candidate)
    if min_version:
        min_version = normalize_version_number(min_version)
    if max_version:
        max_version = normalize_version_number(max_version)
    if min_version and candidate < min_version:
        return False
    if max_version and candidate > max_version:
        return False
    return True