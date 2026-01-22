from reportlab.lib.utils import strTypes
from .flowables import Flowable, _Container, _FindSplitterMixin, _listWrapOn
def getSpaceBefore(self):
    m = self._spaceBefore
    if m is None:
        m = 0
        for F in self.contents:
            m = max(m, _Container.getSpaceBefore(self, F))
    return m