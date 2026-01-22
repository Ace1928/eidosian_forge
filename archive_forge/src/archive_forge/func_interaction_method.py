import json
from collections import namedtuple
import macaroonbakery.bakery as bakery
def interaction_method(self, kind, x):
    """ Checks whether the error is an InteractionRequired error
        that implements the method with the given name, and JSON-unmarshals the
        method-specific data into x by calling its from_dict method
        with the deserialized JSON object.
        @param kind The interaction method kind (string).
        @param x A class with a class method from_dict that returns a new
        instance of the interaction info for the given kind.
        @return The result of x.from_dict.
        """
    if self.info is None or self.code != ERR_INTERACTION_REQUIRED:
        raise InteractionError('not an interaction-required error (code {})'.format(self.code))
    entry = self.info.interaction_methods.get(kind)
    if entry is None:
        raise InteractionMethodNotFound('interaction method {} not found'.format(kind))
    return x.from_dict(entry)