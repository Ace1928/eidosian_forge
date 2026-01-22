from __future__ import annotations
from typing import Any, Callable, Mapping
import numbers
from attrs import evolve, field, frozen
from rpds import HashTrieMap
from jsonschema.exceptions import UndefinedTypeCheck
def redefine_many(self, definitions=()) -> TypeChecker:
    """
        Produce a new checker with the given types redefined.

        Arguments:

            definitions (dict):

                A dictionary mapping types to their checking functions.
        """
    type_checkers = self._type_checkers.update(definitions)
    return evolve(self, type_checkers=type_checkers)