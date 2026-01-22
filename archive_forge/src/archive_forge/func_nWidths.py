from reportlab.lib.utils import strTypes
from .flowables import Flowable, _Container, _FindSplitterMixin, _listWrapOn
def nWidths(self, aW):
    if aW == self._naW:
        return self._nW
    nW = [].append
    widths = self.widths
    s = 0.0
    for i, w in enumerate(widths):
        if isinstance(w, strTypes):
            w = w.strip()
            pc = w.endswith('%')
            if pc:
                w = w[:-1]
            try:
                w = float(w)
            except:
                raise ValueError('%s: nWidths failed with value %r' % (self, widths[i]))
            if pc:
                w = w * 0.01 * aW
        elif not isinstance(w, (float, int)):
            raise ValueError('%s: nWidths failed with value %r' % (self, widths[i]))
        s += w
        nW(w)
    self._naW = aW
    s = aW / s
    self._nW = [w * s for w in nW.__self__]
    return self._nW