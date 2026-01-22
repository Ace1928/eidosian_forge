import eventlet
import __original_module_threading as orig_threading
import threading  # noqa
import sys
from oslo_config import cfg
import oslo_i18n as i18n
from oslo_log import log as logging
from oslo_service import systemd
from heat.cmd import api
from heat.cmd import api_cfn
from heat.cmd import engine
from heat.common import config
from heat.common import messaging
from heat import version
def launch_all(setup_logging=True):
    if setup_logging:
        logging.register_options(cfg.CONF)
    cfg.CONF(project='heat', prog='heat-all', version=version.version_info.version_string())
    if setup_logging:
        logging.setup(cfg.CONF, 'heat-all')
    config.set_config_defaults()
    messaging.setup()
    return _start_service_threads(set(cfg.CONF.heat_all.enabled_services))