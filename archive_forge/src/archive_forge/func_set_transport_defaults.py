import logging
from debtcollector import removals
from oslo_config import cfg
from stevedore import driver
from urllib import parse
from oslo_messaging import exceptions
def set_transport_defaults(control_exchange):
    """Set defaults for messaging transport configuration options.

    :param control_exchange: the default exchange under which topics are scoped
    :type control_exchange: str
    """
    cfg.set_defaults(_transport_opts, control_exchange=control_exchange)