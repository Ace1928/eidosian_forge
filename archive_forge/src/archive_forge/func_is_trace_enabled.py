from oslo_config import cfg
from osprofiler import web
def is_trace_enabled(conf=None):
    if conf is None:
        conf = cfg.CONF
    return conf.profiler.enabled