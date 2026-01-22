import codecs
import io
from .. import tests
from ..progress import ProgressTask
from ..ui.text import TextProgressView
from . import ui_testing
def make_view(self):
    out = ui_testing.StringIOWithEncoding()
    return (out, self.make_view_only(out))