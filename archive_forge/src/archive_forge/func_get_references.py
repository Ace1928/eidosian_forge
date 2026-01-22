import datetime
import re
from collections import namedtuple
from django.db.models.fields.related import RECURSIVE_RELATIONSHIP_CONSTANT
def get_references(state, model_tuple, field_tuple=()):
    """
    Generator of (model_state, name, field, reference) referencing
    provided context.

    If field_tuple is provided only references to this particular field of
    model_tuple will be generated.
    """
    for state_model_tuple, model_state in state.models.items():
        for name, field in model_state.fields.items():
            reference = field_references(state_model_tuple, field, model_tuple, *field_tuple)
            if reference:
                yield (model_state, name, field, reference)