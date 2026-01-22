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
def cleanup_task(graph, service, fn):
    try:
        fn()
    except Exception:
        log = _log.setup_logging('openstack.project_cleanup')
        log.exception('Error in the %s cleanup function' % service)
    finally:
        graph.node_done(service)