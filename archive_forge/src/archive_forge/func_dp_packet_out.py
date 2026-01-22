from os_ken.lib.packet import ethernet
from os_ken.lib.packet import vlan
from os_ken.ofproto import ether
from os_ken.topology import api as topo_api
def dp_packet_out(dp, port_no, data):
    ofproto = dp.ofproto
    ofproto_parser = dp.ofproto_parser
    actions = [ofproto_parser.OFPActionOutput(port_no, ofproto.OFPCML_NO_BUFFER)]
    packet_out = ofproto_parser.OFPPacketOut(dp, 4294967295, ofproto.OFPP_CONTROLLER, actions, data)
    dp.send_msg(packet_out)