import logging
import threading
import enum
from oslo_utils import reflection
from glance_store import exceptions
from glance_store.i18n import _LW
def update_capabilities(self):
    """
        Update dynamic storage capabilities based on current
        driver configuration and backend status when needed.

        As a hook, the function will be triggered in two cases:
        calling once after store driver get configured, it was
        used to update dynamic storage capabilities based on
        current driver configuration, or calling when the
        capabilities checking of an operation failed every time,
        this was used to refresh dynamic storage capabilities
        based on backend status then.

        This function shouldn't raise any exception out.
        """
    LOG.debug("Store %s doesn't support updating dynamic storage capabilities. Please overwrite 'update_capabilities' method of the store to implement updating logics if needed." % reflection.get_class_name(self))