import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_ui_show_user_warning(self):
    from ..bzr.groupcompress_repo import RepositoryFormat2a
    from ..bzr.knitpack_repo import RepositoryFormatKnitPack5
    ui = ui_testing.TextUIFactory()
    remote_fmt = remote.RemoteRepositoryFormat()
    remote_fmt._network_name = RepositoryFormatKnitPack5().network_name()
    ui.show_user_warning('cross_format_fetch', from_format=RepositoryFormat2a(), to_format=remote_fmt)
    self.assertEqual('', ui.stdout.getvalue())
    self.assertContainsRe(ui.stderr.getvalue(), "^Doing on-the-fly conversion from RepositoryFormat2a\\(\\) to RemoteRepositoryFormat\\(_network_name=b?'Bazaar RepositoryFormatKnitPack5 \\(bzr 1.6\\)\\\\n'\\)\\.\nThis may take some time. Upgrade the repositories to the same format for better performance\\.\n$")
    ui = ui_testing.TextUIFactory()
    ui.suppressed_warnings.add('cross_format_fetch')
    ui.show_user_warning('cross_format_fetch', from_format=RepositoryFormat2a(), to_format=remote_fmt)
    self.assertEqual('', ui.stdout.getvalue())
    self.assertEqual('', ui.stderr.getvalue())