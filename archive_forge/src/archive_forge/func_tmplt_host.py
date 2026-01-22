from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_host(verb):
    cmd = 'logging host'
    changed = True
    if verb.get('transport'):
        changed = False
    if verb:
        if verb.get('host'):
            cmd += ' {hostname}'.format(hostname=verb['host'])
        if verb.get('ipv6'):
            cmd += ' ipv6 {ipv6}'.format(ipv6=verb['ipv6'])
        if verb.get('vrf'):
            cmd += ' vrf {vrf}'.format(vrf=verb['vrf'])
        if verb.get('filtered'):
            cmd += ' {filtered}'.format(filtered='filtered')
            changed = True
        if verb.get('xml'):
            cmd += ' {xml}'.format(xml='xml')
            changed = True
        if verb.get('session_id'):
            session_id = verb.get('session_id')
            changed = True
            if session_id.get('text'):
                cmd += ' session-id string {text}'.format(text=session_id['text'])
            elif session_id.get('tag'):
                cmd += ' session-id {tag}'.format(tag=session_id['tag'])
        if verb.get('stream'):
            cmd += ' stream {stream}'.format(stream=verb['stream'])
            changed = True
        if verb.get('sequence_num_session'):
            cmd += ' {sequence_num_session}'.format(sequence_num_session='sequence-num-session')
            changed = True
        if verb.get('discriminator'):
            cmd += ' discriminator {discriminator}'.format(discriminator=verb['discriminator'])
            changed = True
    if not changed:
        cmd = None
    return cmd