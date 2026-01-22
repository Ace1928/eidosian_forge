from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
@classmethod
def to_lightweight_checkout(klass, controldir, reference_location=None):
    """Make a Reconfiguration to convert controldir into a lightweight checkout

        :param controldir: The controldir to reconfigure
        :param bound_location: The location the checkout should be bound to.
        :raise AlreadyLightweightCheckout: if controldir is already a
            lightweight checkout
        """
    reconfiguration = klass(controldir, reference_location)
    reconfiguration._plan_changes(want_tree=True, want_branch=False, want_bound=False, want_reference=True)
    if not reconfiguration.changes_planned():
        raise AlreadyLightweightCheckout(controldir)
    return reconfiguration