from pyomo.common.collections import ComponentMap
from pyomo.core.base import (
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.core.base import TransformationFactory
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.expr import replace_expressions
from pyomo.util.components import rename_components
This method takes the solution in scaled_model and maps it back to
        the original model.

        It will also transform duals and reduced costs if the suffixes
        'dual' and/or 'rc' are present.  The :code:`scaled_model`
        argument must be a model that was already scaled using this
        transformation as it expects data from the transformation to
        perform the back mapping.

        Parameters
        ----------
        scaled_model : Pyomo Model
           The model that was previously scaled with this transformation
        original_model : Pyomo Model
           The original unscaled source model

        