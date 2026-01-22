from reportlab.lib import colors
from ._Colors import ColorTranslator
def set_colour(self, colour):
    """Backwards compatible variant of set_color(self, color) using UK spelling."""
    color = self._colortranslator.translate(colour)
    self.color = color