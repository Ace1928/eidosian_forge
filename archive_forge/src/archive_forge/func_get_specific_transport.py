import eventlet
from oslo_config import cfg
import oslo_messaging
from oslo_messaging.rpc import dispatcher
from osprofiler import profiler
from heat.common import context
def get_specific_transport(url, optional, exmods, is_for_notifications=False):
    try:
        if is_for_notifications:
            return oslo_messaging.get_notification_transport(cfg.CONF, url, allowed_remote_exmods=exmods)
        else:
            return oslo_messaging.get_rpc_transport(cfg.CONF, url, allowed_remote_exmods=exmods)
    except oslo_messaging.InvalidTransportURL as e:
        if not optional or e.url:
            raise
        else:
            return None