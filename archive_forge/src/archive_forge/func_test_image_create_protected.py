import io
from unittest import mock
import requests
from openstack import exceptions
from openstack.image.v2 import _proxy
from openstack.image.v2 import cache as _cache
from openstack.image.v2 import image as _image
from openstack.image.v2 import member as _member
from openstack.image.v2 import metadef_namespace as _metadef_namespace
from openstack.image.v2 import metadef_object as _metadef_object
from openstack.image.v2 import metadef_resource_type as _metadef_resource_type
from openstack.image.v2 import metadef_schema as _metadef_schema
from openstack.image.v2 import schema as _schema
from openstack.image.v2 import service_info as _service_info
from openstack.image.v2 import task as _task
from openstack import proxy as proxy_base
from openstack.tests.unit.image.v2 import test_image as fake_image
from openstack.tests.unit import test_proxy_base
def test_image_create_protected(self):
    self.proxy.find_image = mock.Mock()
    created_image = mock.Mock(spec=_image.Image(id='id'))
    self.proxy._create = mock.Mock()
    self.proxy._create.return_value = created_image
    self.proxy._create.return_value.image_import_methods = []
    created_image.upload = mock.Mock()
    created_image.upload.return_value = FakeResponse(response='', status_code=200)
    properties = {'is_protected': True}
    self.proxy.create_image(name='fake', data='data', container_format='bare', disk_format='raw', **properties)
    args, kwargs = self.proxy._create.call_args
    self.assertEqual(kwargs['is_protected'], True)