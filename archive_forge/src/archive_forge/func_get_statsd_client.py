import copy
import os.path
import typing as ty
from urllib import parse
import warnings
from keystoneauth1 import discover
import keystoneauth1.exceptions.catalog
from keystoneauth1.loading import adapter as ks_load_adap
from keystoneauth1 import session as ks_session
import os_service_types
import requestsexceptions
from openstack import _log
from openstack.config import _util
from openstack.config import defaults as config_defaults
from openstack import exceptions
from openstack import proxy
from openstack import version as openstack_version
from openstack import warnings as os_warnings
def get_statsd_client(self):
    if not statsd:
        if self._statsd_host:
            self.log.warning('StatsD python library is not available. Reporting disabled')
        return None
    statsd_args = {}
    if self._statsd_host:
        statsd_args['host'] = self._statsd_host
    if self._statsd_port:
        statsd_args['port'] = self._statsd_port
    if statsd_args:
        try:
            return statsd.StatsClient(**statsd_args)
        except Exception:
            self.log.warning('Cannot establish connection to statsd')
            return None
    else:
        return None