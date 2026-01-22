from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_host_transport(verb):
    cmd = 'logging host'
    if verb.get('host'):
        cmd += ' {hostname}'.format(hostname=verb['host'])
    if verb.get('ipv6'):
        cmd += ' ipv6 {ipv6}'.format(ipv6=verb['ipv6'])
    if verb.get('vrf'):
        cmd += ' vrf {vrf}'.format(vrf=verb['vrf'])
    if verb.get('transport'):
        transport_type = verb.get('transport')
        prot = None
        if transport_type.get('udp'):
            cmd += ' transport {prot}'.format(prot='udp')
            prot = 'udp'
        elif transport_type.get('tcp'):
            cmd += ' transport {prot}'.format(prot='tcp')
            prot = 'tcp'
        if prot:
            verb = transport_type.get(prot)
            if verb.get('port'):
                cmd += ' port {port}'.format(port=verb['port'])
            if verb.get('audit'):
                cmd += ' {audit}'.format(audit='audit')
            if verb.get('xml'):
                cmd += ' {xml}'.format(xml='xml')
            if verb.get('filtered'):
                cmd += ' {filtered}'.format(filtered='filtered')
            if verb.get('discriminator'):
                cmd += ' discriminator {discriminator}'.format(discriminator=verb['discriminator'])
            if verb.get('stream'):
                cmd += ' stream {stream}'.format(stream=verb['stream'])
            if verb.get('session_id'):
                session_id = verb.get('session_id')
                if session_id.get('text'):
                    cmd += ' session-id string {text}'.format(text=session_id['text'])
                elif session_id.get('tag'):
                    cmd += ' session-id {tag}'.format(tag=session_id['tag'])
            if verb.get('sequence_num_session'):
                cmd += ' {sequence_num_session}'.format(sequence_num_session='sequence-num-session')
    return cmd