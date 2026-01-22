import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch.object(api, 'get_session')
def test_image_property_delete_exception(self, mock_get_session):
    mock_context = mock.MagicMock()
    mock_session = mock_get_session.return_value
    mock_result = mock_session.query.return_value.filter_by.return_value.one.return_value
    mock_result.delete.side_effect = RuntimeError
    self.assertRaises(RuntimeError, api.image_property_delete, mock_context, 'myprop', 'myimage')