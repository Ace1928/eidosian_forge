from ase.gui.i18n import _
import ase.data
import ase.gui.ui as ui
from ase import Atoms
def pywindow(title, callback):
    code = callback()
    if code is None:
        ui.error(_('No Python code'), _('You have not (yet) specified a consistent set of parameters.'))
    else:
        win = ui.Window(title)
        win.add(ui.Text(code))