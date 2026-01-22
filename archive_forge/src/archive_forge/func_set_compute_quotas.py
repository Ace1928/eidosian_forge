import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def set_compute_quotas(self, name_or_id, **kwargs):
    """Set a quota in a project

        :param name_or_id: project name or id
        :param kwargs: key/value pairs of quota name and quota value

        :raises: :class:`~openstack.exceptions.SDKException` if the resource to
            set the quota does not exist.
        """
    proj = self.identity.find_project(name_or_id, ignore_missing=False)
    kwargs['force'] = True
    self.compute.update_quota_set(_qs.QuotaSet(project_id=proj.id), **kwargs)