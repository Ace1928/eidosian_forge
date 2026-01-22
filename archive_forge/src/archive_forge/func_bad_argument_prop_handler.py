import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def bad_argument_prop_handler(revision):
    return {'custom_prop_name': revision.properties['a_prop']}