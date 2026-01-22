from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width
from sympy.utilities.exceptions import sympy_deprecation_warning
def leftslash(self):
    """Precede object by a slash of the proper size.
        """
    height = max(self.baseline, self.height() - 1 - self.baseline) * 2 + 1
    slash = '\n'.join((' ' * (height - i - 1) + xobj('/', 1) + ' ' * i for i in range(height)))
    return self.left(stringPict(slash, height // 2))