from reportlab.rl_config import register_reset
def setFormat(self, counter, format):
    """Specifies that the given counter should use
        the given format henceforth."""
    func = self._formatters[format]
    self._getCounter(counter).setFormatter(func)