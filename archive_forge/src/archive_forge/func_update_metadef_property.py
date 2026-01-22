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
def update_metadef_property(self, metadef_property, metadef_namespace, **attrs):
    """Update a metadef property

        :param metadef_property: The value can be either the name of metadef
            property or an
            :class:`~openstack.image.v2.metadef_property.MetadefProperty`
            instance.
        :param metadef_namespace: The value can be either the name of metadef
            namespace or an
            :class:`~openstack.image.v2.metadef_namespace.MetadefNamespace`
            instance
        :param attrs: The attributes to update on the metadef property
            represented by ``metadef_property``.

        :returns: The updated metadef property
        :rtype: :class:`~openstack.image.v2.metadef_property.MetadefProperty`
        """
    namespace_name = resource.Resource._get_id(metadef_namespace)
    metadef_property = resource.Resource._get_id(metadef_property)
    return self._update(_metadef_property.MetadefProperty, metadef_property, namespace_name=namespace_name, **attrs)