import os
import time
import typing as ty
import warnings
from openstack import exceptions
from openstack.image.v2 import cache as _cache
from openstack.image.v2 import image as _image
from openstack.image.v2 import member as _member
from openstack.image.v2 import metadef_namespace as _metadef_namespace
from openstack.image.v2 import metadef_object as _metadef_object
from openstack.image.v2 import metadef_property as _metadef_property
from openstack.image.v2 import metadef_resource_type as _metadef_resource_type
from openstack.image.v2 import metadef_schema as _metadef_schema
from openstack.image.v2 import schema as _schema
from openstack.image.v2 import service_info as _si
from openstack.image.v2 import task as _task
from openstack import proxy
from openstack import resource
from openstack import utils
from openstack import warnings as os_warnings
def cache_delete_image(self, image, ignore_missing=True):
    """Delete an image from cache.

        :param image: The value can be either the name of an image or a
            :class:`~openstack.image.v2.image.Image`
            instance.
        :param bool ignore_missing: When set to ``False``,
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the metadef namespace does not exist.
        :returns: ``None``
        """
    return self._delete(_cache.Cache, image, ignore_missing=ignore_missing)