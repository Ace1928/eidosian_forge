import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def list_redundant(args=None):
    logging.basicConfig(level=logging.WARN)
    conf = cfg.CONF
    conf.register_cli_opts(ENFORCER_OPTS)
    conf.register_opts(ENFORCER_OPTS)
    conf(args)
    _check_for_namespace_opt(conf)
    _list_redundant(conf.namespace)