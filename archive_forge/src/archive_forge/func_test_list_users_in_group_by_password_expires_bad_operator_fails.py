import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def test_list_users_in_group_by_password_expires_bad_operator_fails(self):
    """Ensure an invalid operator returns a Bad Request.

        GET /groups/{groupid}/users?password_expires_at=
        {invalid_operator}:{timestamp}
        GET /groups/{group_id}/users?password_expires_at=
        {operator}:{timestamp}&{invalid_operator}:{timestamp}

        """
    bad_op_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime), 'bad')
    self.get(bad_op_url, expected_status=http.client.BAD_REQUEST)
    bad_op_url = self._list_users_in_group_by_multiple_password_expires_at(self._format_timestamp(self.starttime), 'lt', self._format_timestamp(self.starttime), 'x')
    self.get(bad_op_url, expected_status=http.client.BAD_REQUEST)