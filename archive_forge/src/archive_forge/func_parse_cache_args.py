import logging
import os
from oslo_config import cfg
from oslo_middleware import cors
from oslo_policy import opts
from oslo_policy import policy
from paste import deploy
from glance.i18n import _
from glance.version import version_info as version
def parse_cache_args(args=None):
    config_files = cfg.find_config_files(project='glance', prog='glance-api')
    config_files.extend(cfg.find_config_files(project='glance', prog='glance-cache'))
    parse_args(args=args, default_config_files=config_files)