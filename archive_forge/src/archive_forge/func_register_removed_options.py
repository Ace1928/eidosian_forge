from glance.i18n import _
from oslo_config import cfg
def register_removed_options():
    cfg.CONF.register_opts(removed_opts)