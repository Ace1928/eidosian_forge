import copy
import functools
import queue
import warnings
import dogpile.cache
import keystoneauth1.exceptions
import keystoneauth1.session
import requests.models
import requestsexceptions
from openstack import _log
from openstack.cloud import _object_store
from openstack.cloud import _utils
from openstack.cloud import meta
import openstack.config
from openstack.config import cloud_region as cloud_region_mod
from openstack import exceptions
from openstack import proxy
from openstack import utils
Cleanup the project resources.

        Cleanup all resources in all services, which provide cleanup methods.

        :param bool dry_run: Cleanup or only list identified resources.
        :param int wait_timeout: Maximum amount of time given to each service
            to comlete the cleanup.
        :param queue status_queue: a threading queue object used to get current
            process status. The queue contain processed resources.
        :param dict filters: Additional filters for the cleanup (only resources
            matching all filters will be deleted, if there are no other
            dependencies).
        :param resource_evaluation_fn: A callback function, which will be
            invoked for each resurce and must return True/False depending on
            whether resource need to be deleted or not.
        :param skip_resources: List of specific resources whose cleanup should
            be skipped.
        