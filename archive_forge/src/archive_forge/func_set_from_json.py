from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
def set_from_json(self, obj, json, *, models=None, setter=None):
    """ Sets the value of this property from a JSON value.

        This method first separately extracts and removes any ``units`` field
        in the JSON, and sets the associated units property directly. The
        remaining JSON is then passed to the superclass ``set_from_json`` to
        be handled.

        Args:
            obj: (HasProps) : instance to set the property value on

            json: (JSON-value) : value to set to the attribute to

            models (dict or None, optional) :
                Mapping of model ids to models (default: None)

                This is needed in cases where the attributes to update also
                have values that have references.

            setter (ClientSession or ServerSession or None, optional) :
                This is used to prevent "boomerang" updates to Bokeh apps.
                (default: None)

                In the context of a Bokeh server application, incoming updates
                to properties will be annotated with the session that is
                doing the updating. This value is propagated through any
                subsequent change notifications that the update triggers.
                The session can compare the event setter to itself, and
                suppress any updates that originate from itself.

        Returns:
            None

        """
    json = self._extract_units(obj, json)
    super().set_from_json(obj, json, setter=setter)