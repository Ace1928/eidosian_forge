import os
import platform
import socket
import sys
import futurist
from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import server
from taskflow import logging
from taskflow import task as t_task
from taskflow.utils import banner
from taskflow.utils import misc
from taskflow.utils import threading_utils as tu
@misc.cachedproperty
def banner(self):
    """A banner that can be useful to display before running."""
    connection_details = self._server.connection_details
    transport = connection_details.transport
    if transport.driver_version:
        transport_driver = '%s v%s' % (transport.driver_name, transport.driver_version)
    else:
        transport_driver = transport.driver_name
    try:
        hostname = socket.getfqdn()
    except socket.error:
        hostname = '???'
    try:
        pid = os.getpid()
    except OSError:
        pid = '???'
    chapters = {'Connection details': {'Driver': transport_driver, 'Exchange': self._exchange, 'Topic': self._topic, 'Transport': transport.driver_type, 'Uri': connection_details.uri}, 'Powered by': {'Executor': reflection.get_class_name(self._executor), 'Thread count': getattr(self._executor, 'max_workers', '???')}, 'Supported endpoints': [str(ep) for ep in self._endpoints], 'System details': {'Hostname': hostname, 'Pid': pid, 'Platform': platform.platform(), 'Python': sys.version.split('\n', 1)[0].strip(), 'Thread id': tu.get_ident()}}
    return banner.make_banner('WBE worker', chapters)