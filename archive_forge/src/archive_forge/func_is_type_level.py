import re
from . import schema
from .search import Search
from .resources import CObject, Project, Projects, Experiment, Experiments
from .uriutil import inv_translate_uri
from .errors import ProgrammingError
def is_type_level(element):
    return element.strip('/') in schema.resources_types and (not is_expand_level(element))