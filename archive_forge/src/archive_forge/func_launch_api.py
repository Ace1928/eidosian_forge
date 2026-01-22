import eventlet
import __original_module_threading as orig_threading
import threading  # noqa
import sys
from oslo_config import cfg
import oslo_i18n as i18n
from oslo_log import log as logging
from oslo_reports import guru_meditation_report as gmr
from oslo_service import systemd
from heat.common import config
from heat.common import messaging
from heat.common import profiler
from heat.common import wsgi
from heat import version
def launch_api(setup_logging=True):
    if setup_logging:
        logging.register_options(CONF)
    CONF(project='heat', prog='heat-api', version=version.version_info.version_string())
    if setup_logging:
        logging.setup(CONF, CONF.prog)
    LOG = logging.getLogger(CONF.prog)
    config.set_config_defaults()
    messaging.setup()
    app = config.load_paste_app()
    port = CONF.heat_api.bind_port
    host = CONF.heat_api.bind_host
    LOG.info('Starting Heat REST API on %(host)s:%(port)s', {'host': host, 'port': port})
    profiler.setup(CONF.prog, host)
    gmr.TextGuruMeditation.setup_autorun(version)
    server = wsgi.Server(CONF.prog, CONF.heat_api)
    server.start(app, default_port=port)
    return server