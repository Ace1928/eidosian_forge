from keystoneauth1 import exceptions as ka_exceptions
from keystoneauth1 import loading as ka_loading
from keystoneclient.v3 import client as ks_client
from oslo_config import cfg
from oslo_log import log as logging
Release keystone resources required for refreshing