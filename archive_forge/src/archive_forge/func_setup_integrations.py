from __future__ import absolute_import
from threading import Lock
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import logger
def setup_integrations(integrations, with_defaults=True, with_auto_enabling_integrations=False):
    """
    Given a list of integration instances, this installs them all.

    When `with_defaults` is set to `True` all default integrations are added
    unless they were already provided before.
    """
    integrations = dict(((integration.identifier, integration) for integration in integrations or ()))
    logger.debug('Setting up integrations (with default = %s)', with_defaults)
    used_as_default_integration = set()
    if with_defaults:
        for integration_cls in iter_default_integrations(with_auto_enabling_integrations):
            if integration_cls.identifier not in integrations:
                instance = integration_cls()
                integrations[instance.identifier] = instance
                used_as_default_integration.add(instance.identifier)
    for identifier, integration in iteritems(integrations):
        with _installer_lock:
            if identifier not in _processed_integrations:
                logger.debug('Setting up previously not enabled integration %s', identifier)
                try:
                    type(integration).setup_once()
                except NotImplementedError:
                    if getattr(integration, 'install', None) is not None:
                        logger.warning('Integration %s: The install method is deprecated. Use `setup_once`.', identifier)
                        integration.install()
                    else:
                        raise
                except DidNotEnable as e:
                    if identifier not in used_as_default_integration:
                        raise
                    logger.debug('Did not enable default integration %s: %s', identifier, e)
                else:
                    _installed_integrations.add(identifier)
                _processed_integrations.add(identifier)
    integrations = {identifier: integration for identifier, integration in iteritems(integrations) if identifier in _installed_integrations}
    for identifier in integrations:
        logger.debug('Enabling integration %s', identifier)
    return integrations