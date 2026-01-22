import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_xml_message(self):
    payload = '<?xml version="1.0" encoding="UTF-8"?>\n<rebuild\n    xmlns="http://docs.openstack.org/compute/api/v1.1"\n    name="foobar"\n    imageRef="http://openstack.example.com/v1.1/32278/images/70a599e0-31e7"\n    accessIPv4="1.2.3.4"\n    accessIPv6="fe80::100"\n    adminPass="seekr3t">\n  <metadata>\n    <meta key="My Server Name">Apache1</meta>\n  </metadata>\n</rebuild>'
    expected = '<?xml version="1.0" encoding="UTF-8"?>\n<rebuild\n    xmlns="http://docs.openstack.org/compute/api/v1.1"\n    name="foobar"\n    imageRef="http://openstack.example.com/v1.1/32278/images/70a599e0-31e7"\n    accessIPv4="1.2.3.4"\n    accessIPv6="fe80::100"\n    adminPass="***">\n  <metadata>\n    <meta key="My Server Name">Apache1</meta>\n  </metadata>\n</rebuild>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    admin_pass="MySecretPass"/>'
    expected = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    admin_pass="***"/>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    admin_password="MySecretPass"/>'
    expected = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    admin_password="***"/>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    password="MySecretPass"/>'
    expected = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    password="***"/>'
    self.assertEqual(expected, strutils.mask_password(payload))