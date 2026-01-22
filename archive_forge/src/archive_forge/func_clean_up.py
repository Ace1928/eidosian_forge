from . import errors, trace, ui, urlutils
from .bzr.remote import RemoteBzrDir
from .controldir import ControlDir, format_registry
from .i18n import gettext
def clean_up(self):
    """Clean-up after a conversion.

        This removes the backup.bzr directory.
        """
    transport = self.transport
    backup_relpath = transport.relpath(self.backup_newpath)
    with ui.ui_factory.nested_progress_bar() as child_pb:
        child_pb.update(gettext('Deleting backup.bzr'))
        transport.delete_tree(backup_relpath)