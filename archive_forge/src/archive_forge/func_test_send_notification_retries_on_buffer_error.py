import testscenarios
from unittest import mock
from confluent_kafka import KafkaException
import oslo_messaging
from oslo_messaging._drivers import impl_kafka as kafka_driver
from oslo_messaging.tests import utils as test_utils
def test_send_notification_retries_on_buffer_error(self):
    target = oslo_messaging.Target(topic='topic_test')
    with mock.patch('confluent_kafka.Producer') as producer:
        fake_producer = mock.MagicMock()
        fake_producer.produce = mock.Mock(side_effect=[BufferError, BufferError, None])
        producer.return_value = fake_producer
        self.driver.send_notification(target, {}, {'payload': ['test_1']}, None, retry=3)
        assert fake_producer.produce.call_count == 3