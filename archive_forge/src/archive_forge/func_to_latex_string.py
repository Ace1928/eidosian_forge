from __future__ import annotations
import re
from fractions import Fraction
def to_latex_string(self) -> str:
    """Generates a LaTeX formatted string. The mode is set by the class variable STRING_MODE, which defaults to
        "SUBSCRIPT". E.g., Fe2O3 is transformed to Fe$_{2}$O$_{3}$. Setting STRING_MODE to "SUPERSCRIPT" creates
        superscript, e.g., Fe2+ becomes Fe^{2+}. The initial string is obtained from the class's __str__ method.

        Returns:
            String for display as in LaTeX with proper superscripts and subscripts.
        """
    str_ = self.to_pretty_string()
    str_ = re.sub('_(\\d+)', '$_{\\1}$', str_)
    str_ = re.sub('\\^([\\d\\+\\-]+)', '$^{\\1}$', str_)
    if self.STRING_MODE == 'SUBSCRIPT':
        return re.sub('([A-Za-z\\(\\)])([\\d\\+\\-\\.]+)', '\\1$_{\\2}$', str_)
    if self.STRING_MODE == 'SUPERSCRIPT':
        return re.sub('([A-Za-z\\(\\)])([\\d\\+\\-\\.]+)', '\\1$^{\\2}$', str_)
    return str_