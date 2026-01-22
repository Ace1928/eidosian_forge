import testscenarios
from unittest import mock
from confluent_kafka import KafkaException
import oslo_messaging
from oslo_messaging._drivers import impl_kafka as kafka_driver
from oslo_messaging.tests import utils as test_utils
def test_listen_for_notifications(self):
    targets_and_priorities = [(oslo_messaging.Target(topic='topic_test_1'), 'sample')]
    with mock.patch('confluent_kafka.Consumer') as consumer:
        self.driver.listen_for_notifications(targets_and_priorities, 'kafka_test', 1000, 10)
        consumer.assert_called_once_with({'bootstrap.servers': '', 'enable.partition.eof': False, 'group.id': 'kafka_test', 'enable.auto.commit': mock.ANY, 'max.partition.fetch.bytes': mock.ANY, 'security.protocol': 'PLAINTEXT', 'sasl.mechanism': 'PLAIN', 'sasl.username': mock.ANY, 'sasl.password': mock.ANY, 'ssl.ca.location': '', 'ssl.certificate.location': '', 'ssl.key.location': '', 'ssl.key.password': '', 'default.topic.config': {'auto.offset.reset': 'latest'}})