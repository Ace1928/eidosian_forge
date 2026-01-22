from io import BytesIO
from .lazy_import import lazy_import
from breezy.i18n import gettext
from . import errors, urlutils
from .trace import note
from .transport import (do_catching_redirections, get_transport,
Extract merge request data.

        :return: tuple with (base_revision_id, target_revision_id, verified)
        