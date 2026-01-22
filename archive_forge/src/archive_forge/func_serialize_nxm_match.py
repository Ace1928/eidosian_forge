import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def serialize_nxm_match(rule, buf, offset):
    old_offset = offset
    if not rule.wc.wildcards & FWW_IN_PORT:
        offset += nxm_put(buf, offset, ofproto_v1_0.NXM_OF_IN_PORT, rule)
    if rule.flow.dl_dst != mac.DONTCARE:
        if rule.wc.dl_dst_mask:
            header = ofproto_v1_0.NXM_OF_ETH_DST_W
        else:
            header = ofproto_v1_0.NXM_OF_ETH_DST
        offset += nxm_put(buf, offset, header, rule)
    if rule.flow.dl_src != mac.DONTCARE:
        if rule.wc.dl_src_mask:
            header = ofproto_v1_0.NXM_OF_ETH_SRC_W
        else:
            header = ofproto_v1_0.NXM_OF_ETH_SRC
        offset += nxm_put(buf, offset, header, rule)
    if not rule.wc.wildcards & FWW_DL_TYPE:
        offset += nxm_put(buf, offset, ofproto_v1_0.NXM_OF_ETH_TYPE, rule)
    if rule.wc.vlan_tci_mask != 0:
        if rule.wc.vlan_tci_mask == UINT16_MAX:
            header = ofproto_v1_0.NXM_OF_VLAN_TCI
        else:
            header = ofproto_v1_0.NXM_OF_VLAN_TCI_W
        offset += nxm_put(buf, offset, header, rule)
    if not rule.wc.wildcards & FWW_NW_DSCP:
        offset += nxm_put(buf, offset, ofproto_v1_0.NXM_OF_IP_TOS, rule)
    if not rule.wc.wildcards & FWW_NW_ECN:
        offset += nxm_put(buf, offset, ofproto_v1_0.NXM_NX_IP_ECN, rule)
    if not rule.wc.wildcards & FWW_NW_TTL:
        offset += nxm_put(buf, offset, ofproto_v1_0.NXM_NX_IP_TTL, rule)
    if not rule.wc.wildcards & FWW_NW_PROTO:
        offset += nxm_put(buf, offset, ofproto_v1_0.NXM_OF_IP_PROTO, rule)
    if not rule.wc.wildcards & FWW_NW_PROTO and rule.flow.nw_proto == inet.IPPROTO_ICMP:
        if rule.wc.tp_src_mask != 0:
            offset += nxm_put(buf, offset, ofproto_v1_0.NXM_OF_ICMP_TYPE, rule)
        if rule.wc.tp_dst_mask != 0:
            offset += nxm_put(buf, offset, ofproto_v1_0.NXM_OF_ICMP_CODE, rule)
    if rule.flow.tp_src != 0:
        if rule.flow.nw_proto == 6:
            if rule.wc.tp_src_mask == UINT16_MAX:
                header = ofproto_v1_0.NXM_OF_TCP_SRC
            else:
                header = ofproto_v1_0.NXM_OF_TCP_SRC_W
        elif rule.flow.nw_proto == 17:
            if rule.wc.tp_src_mask == UINT16_MAX:
                header = ofproto_v1_0.NXM_OF_UDP_SRC
            else:
                header = ofproto_v1_0.NXM_OF_UDP_SRC_W
        else:
            header = 0
        if header != 0:
            offset += nxm_put(buf, offset, header, rule)
    if rule.flow.tp_dst != 0:
        if rule.flow.nw_proto == 6:
            if rule.wc.tp_dst_mask == UINT16_MAX:
                header = ofproto_v1_0.NXM_OF_TCP_DST
            else:
                header = ofproto_v1_0.NXM_OF_TCP_DST_W
        elif rule.flow.nw_proto == 17:
            if rule.wc.tp_dst_mask == UINT16_MAX:
                header = ofproto_v1_0.NXM_OF_UDP_DST
            else:
                header = ofproto_v1_0.NXM_OF_UDP_DST_W
        else:
            header = 0
        if header != 0:
            offset += nxm_put(buf, offset, header, rule)
    if rule.flow.tcp_flags != 0:
        if rule.flow.dl_type in (ether.ETH_TYPE_IP, ether.ETH_TYPE_IPV6):
            if rule.flow.nw_proto == inet.IPPROTO_TCP:
                if rule.wc.tcp_flags_mask == UINT16_MAX:
                    header = ofproto_v1_0.NXM_NX_TCP_FLAGS
                else:
                    header = ofproto_v1_0.NXM_NX_TCP_FLAGS_W
            else:
                header = 0
        else:
            header = 0
        if header != 0:
            offset += nxm_put(buf, offset, header, rule)
    if rule.flow.nw_src != 0:
        if rule.wc.nw_src_mask == UINT32_MAX:
            header = ofproto_v1_0.NXM_OF_IP_SRC
        else:
            header = ofproto_v1_0.NXM_OF_IP_SRC_W
        offset += nxm_put(buf, offset, header, rule)
    if rule.flow.nw_dst != 0:
        if rule.wc.nw_dst_mask == UINT32_MAX:
            header = ofproto_v1_0.NXM_OF_IP_DST
        else:
            header = ofproto_v1_0.NXM_OF_IP_DST_W
        offset += nxm_put(buf, offset, header, rule)
    if not rule.wc.wildcards & FWW_NW_PROTO and rule.flow.nw_proto == inet.IPPROTO_ICMPV6:
        if rule.wc.tp_src_mask != 0:
            offset += nxm_put(buf, offset, ofproto_v1_0.NXM_NX_ICMPV6_TYPE, rule)
        if rule.wc.tp_dst_mask != 0:
            offset += nxm_put(buf, offset, ofproto_v1_0.NXM_NX_ICMPV6_CODE, rule)
    if not rule.wc.wildcards & FWW_IPV6_LABEL:
        offset += nxm_put(buf, offset, ofproto_v1_0.NXM_NX_IPV6_LABEL, rule)
    if len(rule.flow.ipv6_src):
        if len(rule.wc.ipv6_src_mask):
            header = ofproto_v1_0.NXM_NX_IPV6_SRC_W
        else:
            header = ofproto_v1_0.NXM_NX_IPV6_SRC
        offset += nxm_put(buf, offset, header, rule)
    if len(rule.flow.ipv6_dst):
        if len(rule.wc.ipv6_dst_mask):
            header = ofproto_v1_0.NXM_NX_IPV6_DST_W
        else:
            header = ofproto_v1_0.NXM_NX_IPV6_DST
        offset += nxm_put(buf, offset, header, rule)
    if len(rule.flow.nd_target):
        if len(rule.wc.nd_target_mask):
            header = ofproto_v1_0.NXM_NX_ND_TARGET_W
        else:
            header = ofproto_v1_0.NXM_NX_ND_TARGET
        offset += nxm_put(buf, offset, header, rule)
    if rule.flow.arp_spa != 0:
        if rule.wc.arp_spa_mask == UINT32_MAX:
            header = ofproto_v1_0.NXM_OF_ARP_SPA
        else:
            header = ofproto_v1_0.NXM_OF_ARP_SPA_W
        offset += nxm_put(buf, offset, header, rule)
    if rule.flow.arp_tpa != 0:
        if rule.wc.arp_tpa_mask == UINT32_MAX:
            header = ofproto_v1_0.NXM_OF_ARP_TPA
        else:
            header = ofproto_v1_0.NXM_OF_ARP_TPA_W
        offset += nxm_put(buf, offset, header, rule)
    if not rule.wc.wildcards & FWW_ARP_SHA:
        offset += nxm_put(buf, offset, ofproto_v1_0.NXM_NX_ARP_SHA, rule)
    if not rule.wc.wildcards & FWW_ARP_THA:
        offset += nxm_put(buf, offset, ofproto_v1_0.NXM_NX_ARP_THA, rule)
    if rule.flow.nw_frag:
        if rule.wc.nw_frag_mask == FLOW_NW_FRAG_MASK:
            header = ofproto_v1_0.NXM_NX_IP_FRAG
        else:
            header = ofproto_v1_0.NXM_NX_IP_FRAG_W
        offset += nxm_put(buf, offset, header, rule)
    if rule.flow.pkt_mark != 0:
        if rule.wc.pkt_mark_mask == UINT32_MAX:
            header = ofproto_v1_0.NXM_NX_PKT_MARK
        else:
            header = ofproto_v1_0.NXM_NX_PKT_MARK_W
        offset += nxm_put(buf, offset, header, rule)
    if rule.wc.tun_id_mask != 0:
        if rule.wc.tun_id_mask == UINT64_MAX:
            header = ofproto_v1_0.NXM_NX_TUN_ID
        else:
            header = ofproto_v1_0.NXM_NX_TUN_ID_W
        offset += nxm_put(buf, offset, header, rule)
    for i in range(FLOW_N_REGS):
        if rule.wc.regs_bits & 1 << i:
            if rule.wc.regs_mask[i]:
                header = ofproto_v1_0.nxm_nx_reg_w(i)
            else:
                header = ofproto_v1_0.nxm_nx_reg(i)
            offset += nxm_put(buf, offset, header, rule)
    pad_len = round_up(offset) - offset
    msg_pack_into('%dx' % pad_len, buf, offset)
    return offset - old_offset