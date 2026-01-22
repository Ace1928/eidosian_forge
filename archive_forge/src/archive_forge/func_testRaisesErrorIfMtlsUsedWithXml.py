from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib import cloud_api
from gslib import cloud_api_delegator
from gslib import context_config
from gslib import cs_api_map
from gslib.tests import testcase
from gslib.tests.testcase import base
from gslib.tests.util import unittest
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(context_config, 'get_context_config')
def testRaisesErrorIfMtlsUsedWithXml(self, mock_get_context_config):
    mock_context_config = mock.Mock()
    mock_context_config.use_client_certificate = True
    mock_get_context_config.return_value = mock_context_config
    api_map = cs_api_map.GsutilApiMapFactory.GetApiMap(gsutil_api_class_map_factory=cs_api_map.GsutilApiClassMapFactory, support_map={'s3': [cs_api_map.ApiSelector.XML]}, default_map={'s3': cs_api_map.ApiSelector.XML})
    delegator = cloud_api_delegator.CloudApiDelegator(None, api_map, None, None)
    with self.assertRaises(cloud_api.ArgumentException):
        delegator.GetApiSelector(provider='s3')