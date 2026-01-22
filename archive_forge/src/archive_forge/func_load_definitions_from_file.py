import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def load_definitions_from_file(self, definition_file):
    """Load new units definitions from a file

        This method loads additional units definitions from a user
        specified definition file. An example of a definitions file
        can be found at:
        https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

        If we have a file called ``my_additional_units.txt`` with the
        following lines::

            USD = [currency]

        Then we can add this to the container with:

        .. doctest::
            :skipif: not pint_available
            :hide:

            # Get a local units object (to avoid duplicate registration
            # with the example in load_definitions_from_strings)
            >>> import pyomo.core.base.units_container as _units
            >>> u = _units.PyomoUnitsContainer()
            >>> with open('my_additional_units.txt', 'w') as FILE:
            ...     tmp = FILE.write("USD = [currency]\\n")

        .. doctest::
            :skipif: not pint_available

            >>> u.load_definitions_from_file('my_additional_units.txt')
            >>> print(u.USD)
            USD

        .. doctest::
            :skipif: not pint_available
            :hide:

            # Clean up the file we just created
            >>> import os
            >>> os.remove('my_additional_units.txt')

        """
    self._pint_registry.load_definitions(definition_file)
    self._pint_dimensionless = self._pint_registry.dimensionless