import codecs
import io
from .. import tests
from ..progress import ProgressTask
from ..ui.text import TextProgressView
from . import ui_testing
def test_render_progress_nested(self):
    """Tasks proportionally contribute to overall progress"""
    out, view = self.make_view()
    task = self.make_task(None, view, 'reticulating splines', 0, 2)
    task2 = self.make_task(task, view, 'stage2', 1, 2)
    view.show_progress(task2)
    view.enable_bar = True
    self.assertEqual('[####-               ] reticulating splines:stage2 1/2                         ', view._render_line())
    task2.update('stage2', 2, 2)
    self.assertEqual('[#########\\          ] reticulating splines:stage2 2/2                         ', view._render_line())