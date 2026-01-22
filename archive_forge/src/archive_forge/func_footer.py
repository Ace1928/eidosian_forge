from typing import Optional
from pyquil.latex._diagram import DiagramBuilder, DiagramSettings
from pyquil.quil import Program
def footer() -> str:
    """
    Return the footer of the LaTeX document.

    :return: LaTeX document footer.
    """
    return '\\end{tikzcd}\n\\end{document}'