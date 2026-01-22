from pyomo.contrib.pynumero.interfaces.nlp import NLP, ExtendedNLP
import numpy as np
import scipy.sparse as sp

        This class takes an NLP that depends on a set of primals (original
        space) and converts it to an NLP that depends on a reordered set of
        primals (projected space).

        This will impact all the returned items associated with primal
        variables. E.g., the gradient will be in the new primals ordering
        instead of the original primals ordering.

        Note also that this can include additional primal variables not
        in the original NLP, or can exclude primal variables that were
        in the original NLP.

        Parameters
        ----------
        original_nlp : NLP-like
            The original NLP object that implements the NLP interface

        primals_ordering: list
           List of strings indicating the desired primal variable
           ordering for this NLP. The list can contain new variables
           that are not in the original NLP, thereby expanding the
           space of the primal variables.
        