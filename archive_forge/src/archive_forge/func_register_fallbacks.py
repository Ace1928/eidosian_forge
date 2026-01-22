import itertools
from yaql.language import contexts
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
def register_fallbacks(context):
    context.register_function(get_property)