from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_from_params_map(params_map, data):
    ret_data = {}
    for want_key, config_key in params_map.items():
        tmp_data = {}
        for key, val in data.items():
            if key == 'config':
                for k, v in val.items():
                    if k == config_key:
                        val_data = val[config_key]
                        ret_data.update({want_key: val_data})
                        if config_key == 'afi-safi-name':
                            ret_data.pop(want_key)
                            for type_k, type_val in afi_safi_types_map.items():
                                if type_k == val_data:
                                    afi_safi = type_val.split('_')
                                    val_data = afi_safi[0]
                                    ret_data.update({'safi': afi_safi[1]})
                                    ret_data.update({want_key: val_data})
                                    break
            elif key == 'timers' and ('config' in val or 'state' in val):
                tmp = {}
                if key in ret_data:
                    tmp = ret_data[key]
                cfg = val['config'] if 'config' in val else val['state']
                for k, v in cfg.items():
                    if k == config_key:
                        if k != 'minimum-advertisement-interval':
                            tmp.update({want_key: cfg[config_key]})
                        else:
                            ret_data.update({want_key: cfg[config_key]})
                if tmp:
                    ret_data.update({key: tmp})
            elif isinstance(config_key, list):
                i = 0
                if key == config_key[0]:
                    if key == 'afi-safi':
                        cfg_data = config_key[1]
                        for itm in afi_safi_types_map:
                            if cfg_data in itm:
                                afi_safi = itm[cfg_data].split('_')
                                cfg_data = afi_safi[0]
                                ret_data.update({'safi': afi_safi[1]})
                                ret_data.update({want_key: cfg_data})
                                break
                    else:
                        cfg_data = {key: val}
                        for cfg_key in config_key:
                            if cfg_key == 'config':
                                continue
                            new_data = None
                            if cfg_key in cfg_data:
                                new_data = cfg_data[cfg_key]
                            elif isinstance(cfg_data, dict) and 'config' in cfg_data:
                                if cfg_key in cfg_data['config']:
                                    new_data = cfg_data['config'][cfg_key]
                            if new_data is not None:
                                cfg_data = new_data
                            else:
                                break
                        else:
                            ret_data.update({want_key: cfg_data})
            elif key == config_key and val:
                if config_key != 'afi-safi-name' and config_key != 'timers':
                    cfg_data = val
                    ret_data.update({want_key: cfg_data})
    return ret_data