import tempfile
from yaql.cli.cli_functions import load_data
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
@specs.parameter('arg', yaqltypes.Iterator())
def baz1(arg):
    return arg is None