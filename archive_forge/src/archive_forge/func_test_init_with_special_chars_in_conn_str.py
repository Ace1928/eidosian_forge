import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
@mock.patch.object(loginsight, 'LogInsightClient')
def test_init_with_special_chars_in_conn_str(self, client_class):
    client = mock.Mock()
    client_class.return_value = client
    loginsight.LogInsightDriver('loginsight://username:p%40ssword@host')
    client_class.assert_called_once_with('host', 'username', 'p@ssword')
    client.login.assert_called_once_with()