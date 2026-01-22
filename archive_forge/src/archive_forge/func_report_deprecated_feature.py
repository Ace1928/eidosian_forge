import functools
import inspect
import logging
from oslo_config import cfg
from oslo_log._i18n import _
def report_deprecated_feature(logger, msg, *args, **kwargs):
    """Call this function when a deprecated feature is used.

    If the system is configured for fatal deprecations then the message
    is logged at the 'critical' level and :class:`DeprecatedConfig` will
    be raised.

    Otherwise, the message will be logged (once) at the 'warn' level.

    :raises: :class:`DeprecatedConfig` if the system is configured for
             fatal deprecations.
    """
    stdmsg = _('Deprecated: %s') % msg
    register_options()
    if CONF.fatal_deprecations:
        logger.critical(stdmsg, *args, **kwargs)
        raise DeprecatedConfig(msg=stdmsg)
    sent_args = _deprecated_messages_sent.setdefault(msg, list())
    if args in sent_args:
        return
    sent_args.append(args)
    logger.warning(stdmsg, *args, **kwargs)