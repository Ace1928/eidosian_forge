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
def delete_all_metadef_properties(self, metadef_namespace):
    """Delete all metadata definitions property inside a specific namespace.

        :param metadef_namespace: The value can be either the name of a metadef
            namespace or a
            :class:`~openstack.image.v2.metadef_namespace.MetadefNamespace`
            instance.

        :returns: ``None``
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            resource can be found.
        """
    namespace = self._get_resource(_metadef_namespace.MetadefNamespace, metadef_namespace)
    return namespace.delete_all_properties(self)