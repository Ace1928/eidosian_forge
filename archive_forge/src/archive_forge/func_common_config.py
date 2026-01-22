from __future__ import annotations
from queue import Empty
from kombu.transport import virtual
from kombu.utils import cached_property
from kombu.utils.encoding import str_to_bytes
from kombu.utils.json import dumps, loads
from kombu.log import get_logger
@cached_property
def common_config(self):
    conninfo = self.connection.client
    config = {'bootstrap.servers': f'{conninfo.hostname}:{int(conninfo.port) or DEFAULT_PORT}'}
    security_protocol = self.options.get('security_protocol', 'plaintext')
    if security_protocol.lower() != 'plaintext':
        config.update({'security.protocol': security_protocol, 'sasl.username': conninfo.userid, 'sasl.password': conninfo.password, 'sasl.mechanism': self.options.get('sasl_mechanism')})
    config.update(self.options.get('kafka_common_config') or {})
    return config