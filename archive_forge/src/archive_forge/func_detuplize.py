import os.path
import json
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.dataportal.factory import DataManagerFactory
def detuplize(d, sort=False):
    if type(d) in (list, tuple, set):
        ans = []
        for item in d:
            if type(item) in (list, tuple, set):
                ans.append(list(item))
            else:
                ans.append(item)
        if sort:
            return sorted(ans)
        return ans
    elif None in d:
        return d[None]
    else:
        ans = []
        for k, v in d.items():
            if type(k) is tuple:
                ans.append({'index': list(k), 'value': v})
            else:
                ans.append({'index': k, 'value': v})
        if sort:
            return sorted(ans, key=lambda x: x['value'])
        return ans