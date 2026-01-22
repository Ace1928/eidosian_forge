from __future__ import (absolute_import, division, print_function)
import inspect
import os
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from ansible_collections.ovirt.ovirt.plugins.module_utils.cloud import CloudRetry
from ansible_collections.ovirt.ovirt.plugins.module_utils.version import ComparableVersion
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common._collections_compat import Mapping
def follow_link(connection, link):
    """
    This method returns the entity of the element which link points to.

    :param connection: connection to the Python SDK
    :param link: link of the entity
    :return: entity which link points to
    """
    if link:
        return connection.follow_link(link)
    else:
        return None