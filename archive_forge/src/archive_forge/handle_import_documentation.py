from pythran.passmanager import Transformation
from pythran.tables import MODULES, pythran_ward
from pythran.syntax import PythranSyntaxError
import gast as ast
import logging
import os
This pass handle user-defined import, mangling name for function from
    other modules and include them in the current module, patching all call
    site accordingly.
    