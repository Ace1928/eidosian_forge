import inspect
import jmespath
from botocore.compat import six
def get_identifier_values_for_example(identifier_names):
    example_values = ["'%s'" % identifier for identifier in identifier_names]
    return ','.join(example_values)