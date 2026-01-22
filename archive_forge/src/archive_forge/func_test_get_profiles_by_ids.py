import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_profiles_by_ids(self):
    pbm_service = mock.Mock()
    session = mock.Mock(pbm=pbm_service)
    profiles = mock.sentinel.profiles
    session.invoke_api.return_value = profiles
    profile_ids = mock.sentinel.profile_ids
    ret = pbm.get_profiles_by_ids(session, profile_ids)
    self.assertEqual(profiles, ret)
    session.invoke_api.assert_called_once_with(pbm_service, 'PbmRetrieveContent', pbm_service.service_content.profileManager, profileIds=profile_ids)