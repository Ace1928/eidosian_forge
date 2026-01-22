from collections import OrderedDict
from itertools import chain
def check_aliases(names, d, k_position=-1, v_position=0):
    for name in [n for n in names]:
        for k, v in d.items():
            v = [v] if not isinstance(v, list) else v
            for vname in v:
                if name == vname and k not in names:
                    if k_position == -2:
                        names.append(k)
                    else:
                        names.insert(k_position, k)
                if name == k and vname not in names:
                    if v_position == -2:
                        names.append(vname)
                    else:
                        names.insert(v_position, vname)
    return names