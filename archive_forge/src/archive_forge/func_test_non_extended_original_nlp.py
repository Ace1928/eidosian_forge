import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def test_non_extended_original_nlp(self):
    m, nlp, proj_nlp = self._get_nlps()
    proj_nlp = ProjectedNLP(nlp, ['x[0]', 'x[1]', 'x[2]'])
    msg = 'Original NLP must be an instance of ExtendedNLP'
    with self.assertRaisesRegex(TypeError, msg):
        proj_ext_nlp = ProjectedExtendedNLP(proj_nlp, ['x[1]', 'x[0]'])