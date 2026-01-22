import logging
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.lib import ofctl_utils
def instructions_to_str(instructions):
    s = []
    for i in instructions:
        v = i.to_jsondict()[i.__class__.__name__]
        t = UTIL.ofp_instruction_type_to_user(v['type'])
        inst_type = t if t != v['type'] else 'UNKNOWN'
        if isinstance(i, ofproto_v1_4_parser.OFPInstructionActions):
            acts = []
            for a in i.actions:
                acts.append(action_to_str(a))
            v['type'] = inst_type
            v['actions'] = acts
            s.append(v)
        else:
            v['type'] = inst_type
            s.append(v)
    return s