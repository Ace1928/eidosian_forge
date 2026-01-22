import gettext
import fixtures
from oslo_i18n import _lazy
from oslo_i18n import _message
def immediate(self, msg):
    """Return a string as though it had been translated immediately.

        :param msg: Input message string. May optionally include
                    positional or named string interpolation markers.
        :type msg: str or unicode

        """
    return str(msg)