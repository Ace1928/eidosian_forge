from reportlab.lib.utils import strTypes
from .flowables import Flowable, _Container, _FindSplitterMixin, _listWrapOn
def getSpaceAfter(self):
    m = self._spaceAfter
    if m is None:
        m = 0
        for F in self.contents:
            m = max(m, _Container.getSpaceAfter(self, F))
    return m