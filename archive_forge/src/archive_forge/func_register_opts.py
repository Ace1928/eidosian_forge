from oslo_config import cfg
from keystone.conf import utils
def register_opts(conf):
    conf.register_opts(ALL_OPTS, group=GROUP_NAME)