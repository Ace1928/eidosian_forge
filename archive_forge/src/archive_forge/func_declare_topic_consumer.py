import logging
import threading
import confluent_kafka
from confluent_kafka import KafkaException
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers.kafka_driver import kafka_options
def declare_topic_consumer(self, topics, group=None):
    conf = {'bootstrap.servers': ','.join(self.hostaddrs), 'group.id': group or self.group_id, 'enable.auto.commit': self.use_auto_commit, 'max.partition.fetch.bytes': self.max_fetch_bytes, 'security.protocol': self.security_protocol, 'sasl.mechanism': self.sasl_mechanism, 'sasl.username': self.username, 'sasl.password': self.password, 'ssl.ca.location': self.ssl_cafile, 'ssl.certificate.location': self.ssl_client_cert_file, 'ssl.key.location': self.ssl_client_key_file, 'ssl.key.password': self.ssl_client_key_password, 'enable.partition.eof': False, 'default.topic.config': {'auto.offset.reset': 'latest'}}
    LOG.debug('Subscribing to %s as %s', topics, group or self.group_id)
    self.consumer = confluent_kafka.Consumer(conf)
    self.consumer.subscribe(topics, on_assign=self.on_assign, on_revoke=self.on_revoke)