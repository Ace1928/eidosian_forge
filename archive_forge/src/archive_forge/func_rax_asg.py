from __future__ import absolute_import, division, print_function
import base64
import json
import os
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (
from ansible.module_utils.six import string_types
def rax_asg(module, cooldown=300, disk_config=None, files=None, flavor=None, image=None, key_name=None, loadbalancers=None, meta=None, min_entities=0, max_entities=0, name=None, networks=None, server_name=None, state='present', user_data=None, config_drive=False, wait=True, wait_timeout=300):
    files = {} if files is None else files
    loadbalancers = [] if loadbalancers is None else loadbalancers
    meta = {} if meta is None else meta
    networks = [] if networks is None else networks
    changed = False
    au = pyrax.autoscale
    if not au:
        module.fail_json(msg='Failed to instantiate clients. This typically indicates an invalid region or an incorrectly capitalized region name.')
    if user_data:
        config_drive = True
    if user_data and os.path.isfile(user_data):
        try:
            f = open(user_data)
            user_data = f.read()
            f.close()
        except Exception as e:
            module.fail_json(msg='Failed to load %s' % user_data)
    if state == 'present':
        if meta:
            for k, v in meta.items():
                if isinstance(v, list):
                    meta[k] = ','.join(['%s' % i for i in v])
                elif isinstance(v, dict):
                    meta[k] = json.dumps(v)
                elif not isinstance(v, string_types):
                    meta[k] = '%s' % v
        if image:
            image = rax_find_image(module, pyrax, image)
        nics = []
        if networks:
            for network in networks:
                nics.extend(rax_find_network(module, pyrax, network))
            for nic in nics:
                if nic.get('net-id'):
                    nic.update(uuid=nic['net-id'])
                    del nic['net-id']
        personality = rax_scaling_group_personality_file(module, files)
        lbs = []
        if loadbalancers:
            for lb in loadbalancers:
                try:
                    lb_id = int(lb.get('id'))
                except (ValueError, TypeError):
                    module.fail_json(msg='Load balancer ID is not an integer: %s' % lb.get('id'))
                try:
                    port = int(lb.get('port'))
                except (ValueError, TypeError):
                    module.fail_json(msg='Load balancer port is not an integer: %s' % lb.get('port'))
                if not lb_id or not port:
                    continue
                lbs.append((lb_id, port))
        try:
            sg = au.find(name=name)
        except pyrax.exceptions.NoUniqueMatch as e:
            module.fail_json(msg='%s' % e.message)
        except pyrax.exceptions.NotFound:
            try:
                sg = au.create(name, cooldown=cooldown, min_entities=min_entities, max_entities=max_entities, launch_config_type='launch_server', server_name=server_name, image=image, flavor=flavor, disk_config=disk_config, metadata=meta, personality=personality, networks=nics, load_balancers=lbs, key_name=key_name, config_drive=config_drive, user_data=user_data)
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
        if not changed:
            group_args = {}
            if cooldown != sg.cooldown:
                group_args['cooldown'] = cooldown
            if min_entities != sg.min_entities:
                group_args['min_entities'] = min_entities
            if max_entities != sg.max_entities:
                group_args['max_entities'] = max_entities
            if group_args:
                changed = True
                sg.update(**group_args)
            lc = sg.get_launch_config()
            lc_args = {}
            if server_name != lc.get('name'):
                lc_args['server_name'] = server_name
            if image != lc.get('image'):
                lc_args['image'] = image
            if flavor != lc.get('flavor'):
                lc_args['flavor'] = flavor
            disk_config = disk_config or 'AUTO'
            if (disk_config or lc.get('disk_config')) and disk_config != lc.get('disk_config', 'AUTO'):
                lc_args['disk_config'] = disk_config
            if (meta or lc.get('meta')) and meta != lc.get('metadata'):
                lc_args['metadata'] = meta
            test_personality = []
            for p in personality:
                test_personality.append({'path': p['path'], 'contents': base64.b64encode(p['contents'])})
            if (test_personality or lc.get('personality')) and test_personality != lc.get('personality'):
                lc_args['personality'] = personality
            if nics != lc.get('networks'):
                lc_args['networks'] = nics
            if lbs != lc.get('load_balancers'):
                lc_args['load_balancers'] = sg.manager._resolve_lbs(lbs)
            if key_name != lc.get('key_name'):
                lc_args['key_name'] = key_name
            if config_drive != lc.get('config_drive', False):
                lc_args['config_drive'] = config_drive
            if user_data and base64.b64encode(user_data) != lc.get('user_data'):
                lc_args['user_data'] = user_data
            if lc_args:
                if 'flavor' not in lc_args:
                    lc_args['flavor'] = lc.get('flavor')
                changed = True
                sg.update_launch_config(**lc_args)
            sg.get()
        if wait:
            end_time = time.time() + wait_timeout
            infinite = wait_timeout == 0
            while infinite or time.time() < end_time:
                state = sg.get_state()
                if state['pending_capacity'] == 0:
                    break
                time.sleep(5)
        module.exit_json(changed=changed, autoscale_group=rax_to_dict(sg))
    else:
        try:
            sg = au.find(name=name)
            sg.delete()
            changed = True
        except pyrax.exceptions.NotFound as e:
            sg = {}
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
        module.exit_json(changed=changed, autoscale_group=rax_to_dict(sg))