from reportlab.rl_config import register_reset
def thisf(self, counter=None):
    if not counter:
        counter = self._defaultCounter
    return self._getCounter(counter).thisf()