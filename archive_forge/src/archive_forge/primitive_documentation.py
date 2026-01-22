from __future__ import annotations
import logging # isort:skip
import numbers
import numpy as np
from ._sphinx import property_link, register_type_link
from .bases import Init, PrimitiveProperty
 Accept string values.

    Args:
        default (string, optional) :
            A default value for attributes created from this property to have.

        help (str or None, optional) :
            A documentation string for this property. It will be automatically
            used by the :ref:`bokeh.sphinxext.bokeh_prop` extension when
            generating Spinx documentation. (default: None)

    Example:

        .. code-block:: python

            >>> class StringModel(HasProps):
            ...     prop = String()
            ...

            >>> m = StringModel()

            >>> m.prop = "foo"

            >>> m.prop = 10.3       # ValueError !!

            >>> m.prop = [1, 2, 3]  # ValueError !!

    