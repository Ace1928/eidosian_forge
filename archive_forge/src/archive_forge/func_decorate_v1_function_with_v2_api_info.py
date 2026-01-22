from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import encoding
def decorate_v1_function_with_v2_api_info(v1_func, v2_func):
    """Decorate gen1 function in v1 API format with additional info from its v2 API format.

  Currently only the `environment` and `upgradeInfo` fields are copied over.

  Args:
    v1_func: A gen1 function retrieved from v1 API.
    v2_func: The same gen1 function but as returned by the v2 API.

  Returns:
    The given Gen1 function encoded as a dict in the v1 format but with the
      `upgradeInfo` and `environment` properties from the v2 format added.
  """
    v1_dict = encoding.MessageToDict(v1_func)
    if v2_func and v2_func.environment:
        v1_dict['environment'] = str(v2_func.environment)
    if v2_func and v2_func.upgradeInfo:
        v1_dict['upgradeInfo'] = encoding.MessageToDict(v2_func.upgradeInfo)
    return v1_dict