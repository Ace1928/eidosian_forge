from unittest import mock
from oslo_config import cfg
from osprofiler.drivers import jaeger
from osprofiler import opts
from osprofiler.tests import test
from jaeger_client import Config
@mock.patch('jaeger_client.span.Span')
@mock.patch('time.time')
def test_notify_stop(self, mock_time, mock_span):
    fake_time = 1525416065.5958152
    mock_time.return_value = fake_time
    span = mock_span()
    self.driver.spans.append(mock_span())
    self.driver.notify(self.payload_stop)
    mock_time.assert_called_once()
    mock_time.reset_mock()
    span.finish.assert_called_once_with(finish_time=fake_time)