import testscenarios
from unittest import mock
from confluent_kafka import KafkaException
import oslo_messaging
from oslo_messaging._drivers import impl_kafka as kafka_driver
from oslo_messaging.tests import utils as test_utils
Unit Test cases to test the kafka driver
    