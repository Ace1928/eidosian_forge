import inspect
import jmespath
from botocore.compat import six
def get_identifier_args_for_signature(identifier_names):
    return ','.join(identifier_names)