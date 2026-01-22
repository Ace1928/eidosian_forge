import configparser
import logging
import logging.config
import logging.handlers
import os
import platform
import sys
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_utils import units
from oslo_log._i18n import _
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
def tempest_set_log_file(filename):
    """Provide an API for tempest to set the logging filename.

    .. warning:: Only Tempest should use this function.

    We don't want applications to set a default log file, so we don't
    want this in set_defaults(). Because tempest doesn't use a
    configuration file we don't have another convenient way to safely
    set the log file default.

    """
    cfg.set_defaults(_options.logging_cli_opts, log_file=filename)