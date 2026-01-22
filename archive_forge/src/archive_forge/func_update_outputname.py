from ase.gui.i18n import _
import ase.gui.ui as ui
from ase.io.pov import write_pov, get_bondpairs
from os import unlink
import numpy as np
def update_outputname(self):
    tokens = [self.basename_widget.value]
    movielen = len(self.gui.images)
    if movielen > 1:
        ndigits = len(str(movielen))
        token = ('{:0' + str(ndigits) + 'd}').format(self.gui.frame)
        tokens.append(token)
    tokens.append('pov')
    fname = '.'.join(tokens)
    self.outputname_widget.text = fname
    return fname