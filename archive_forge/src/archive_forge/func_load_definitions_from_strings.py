import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def load_definitions_from_strings(self, definition_string_list):
    """Load new units definitions from a string

        This method loads additional units definitions from a list of
        strings (one for each line). An example of the definitions
        strings can be found at:
        https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

        For example, to add the currency dimension and US dollars as a
        unit, use

        .. doctest::
            :skipif: not pint_available
            :hide:

            # get a local units object (to avoid duplicate registration
            # with the example in load_definitions_from_strings)
            >>> import pint
            >>> import pyomo.core.base.units_container as _units
            >>> u = _units.PyomoUnitsContainer()

        .. doctest::
            :skipif: not pint_available

            >>> u.load_definitions_from_strings(['USD = [currency]'])
            >>> print(u.USD)
            USD

        """
    self._pint_registry.load_definitions(definition_string_list)