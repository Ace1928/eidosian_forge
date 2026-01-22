import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def trivial_custom_prop_handler(revision):
    return {'test_prop': 'test_value'}