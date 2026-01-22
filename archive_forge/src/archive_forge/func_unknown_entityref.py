import _markupbase
import re
def unknown_entityref(self, ref):
    self.flush()
    print('*** unknown entity ref: &' + ref + ';')