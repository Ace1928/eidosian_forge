from reportlab.rl_config import register_reset
def chain(self, parent, child):
    p = self._getCounter(parent)
    c = self._getCounter(child)
    p.chain(c)