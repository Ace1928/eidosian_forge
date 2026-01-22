import re
import os_ken.exception
from os_ken.lib.ofctl_utils import str_to_int
from os_ken.ofproto import nicira_ext
@classmethod
def ct(cls, ofproto, action_str):
    str_to_port = {'ftp': 21, 'tftp': 69}
    flags = 0
    zone_src = ''
    zone_ofs_nbits = 0
    recirc_table = nicira_ext.NX_CT_RECIRC_NONE
    alg = 0
    ct_actions = []
    if len(action_str) > 2:
        if not action_str.startswith('ct(') or action_str[-1] != ')':
            raise os_ken.exception.OFPInvalidActionString(action_str=action_str)
        rest = tokenize_ofp_instruction_arg(action_str[len('ct('):-1])
    else:
        rest = []
    for arg in rest:
        if arg == 'commit':
            flags |= nicira_ext.NX_CT_F_COMMIT
            rest = rest[len('commit'):]
        elif arg == 'force':
            flags |= nicira_ext.NX_CT_F_FORCE
        elif arg.startswith('exec('):
            ct_actions = ofp_instruction_from_str(ofproto, arg[len('exec('):-1])
        else:
            try:
                k, v = arg.split('=', 1)
                if k == 'table':
                    recirc_table = str_to_int(v)
                elif k == 'zone':
                    m = re.search('\\[(\\d*)\\.\\.(\\d*)\\]', v)
                    if m:
                        zone_ofs_nbits = nicira_ext.ofs_nbits(int(m.group(1)), int(m.group(2)))
                        zone_src = nxm_field_name_to_os_ken(v[:m.start(0)])
                    else:
                        zone_ofs_nbits = str_to_int(v)
                elif k == 'alg':
                    alg = str_to_port[arg[len('alg='):]]
            except Exception:
                raise os_ken.exception.OFPInvalidActionString(action_str=action_str)
    return dict(NXActionCT={'flags': flags, 'zone_src': zone_src, 'zone_ofs_nbits': zone_ofs_nbits, 'recirc_table': recirc_table, 'alg': alg, 'actions': ct_actions})