from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def create_spec(self, **kwargs):
    spec = '{name} {{\n        {options}\n        }}\n        '
    return spec.format(name=self.spec_name, options=self.options)