from io import StringIO
from pyomo.common.gc_manager import PauseGC
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
import logging

        Write a model in the GAMS modeling language format.

        Keyword Arguments
        -----------------
        output_filename: str
            Name of file to write GAMS model to. Optionally pass a file-like
            stream and the model will be written to that instead.
        io_options: dict
            - warmstart=True
                Warmstart by initializing model's variables to their values.
            - symbolic_solver_labels=False
                Use full Pyomo component names rather than
                shortened symbols (slower, but useful for debugging).
            - labeler=None
                Custom labeler. Incompatible with symbolic_solver_labels.
            - solver=None
                If None, GAMS will use default solver for model type.
            - mtype=None
                Model type. If None, will chose from lp, nlp, mip, and minlp.
            - add_options=None
                List of additional lines to write directly
                into model file before the solve statement.
                For model attributes, <model name> is GAMS_MODEL.
            - skip_trivial_constraints=False
                Skip writing constraints whose body section is fixed.
            - output_fixed_variables=False
                If True, output fixed variables as variables; otherwise,
                output numeric value.
            - file_determinism=1
                | How much effort do we want to put into ensuring the
                | GAMS file is written deterministically for a Pyomo model:
                |     0 : None
                |     1 : sort keys of indexed components (default)
                |     2 : sort keys AND sort names (over declaration order)
            - put_results=None
                Filename for optionally writing solution values and
                marginals.  If put_results_format is 'gdx', then GAMS
                will write solution values and marginals to
                GAMS_MODEL_p.gdx and solver statuses to
                {put_results}_s.gdx.  If put_results_format is 'dat',
                then solution values and marginals are written to
                (put_results).dat, and solver statuses to (put_results +
                'stat').dat.
            - put_results_format='gdx'
                Format used for put_results, one of 'gdx', 'dat'.

        