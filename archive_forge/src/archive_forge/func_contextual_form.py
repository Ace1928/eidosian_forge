import gettext
import os
from oslo_i18n import _lazy
from oslo_i18n import _locale
from oslo_i18n import _message
@property
def contextual_form(self):
    """The contextual translation function.

        The returned function takes two values, the context of
        the unicode string, the unicode string to be translated.

        .. versionadded:: 2.1.0

        """
    return self._make_contextual_translation_func()