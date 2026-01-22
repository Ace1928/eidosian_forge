from sympy.printing.mathml import mathml
from sympy.utilities.mathml import c2p
import tempfile
import subprocess
Print to Gtkmathview, a gtk widget capable of rendering MathML.

    Needs libgtkmathview-bin