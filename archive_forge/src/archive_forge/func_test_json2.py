import logging
import os
import sys
import unittest
from os_ken.utils import binary_str
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_json2(self):
    withdrawn_routes = [bgp.BGPWithdrawnRoute(length=0, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=1, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=3, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=7, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=32, addr='192.0.2.13')]
    mp_nlri = [bgp.LabelledVPNIPAddrPrefix(24, '192.0.9.0', route_dist='100:100', labels=[1, 2, 3]), bgp.LabelledVPNIPAddrPrefix(26, '192.0.10.192', route_dist='10.0.0.1:10000', labels=[5, 6, 7, 8])]
    mp_nlri2 = [bgp.LabelledIPAddrPrefix(24, '192.168.0.0', labels=[1, 2, 3])]
    mp_nlri_v6 = [bgp.LabelledVPNIP6AddrPrefix(64, '2001:db8:1111::', route_dist='200:200', labels=[1, 2, 3]), bgp.LabelledVPNIP6AddrPrefix(64, '2001:db8:2222::', route_dist='10.0.0.1:10000', labels=[5, 6, 7, 8])]
    mp_nlri2_v6 = [bgp.LabelledIP6AddrPrefix(64, '2001:db8:3333::', labels=[1, 2, 3])]
    communities = [bgp.BGP_COMMUNITY_NO_EXPORT, bgp.BGP_COMMUNITY_NO_ADVERTISE]
    ecommunities = [bgp.BGPTwoOctetAsSpecificExtendedCommunity(subtype=1, as_number=65500, local_administrator=3908892843), bgp.BGPFourOctetAsSpecificExtendedCommunity(subtype=2, as_number=10000000, local_administrator=59876), bgp.BGPIPv4AddressSpecificExtendedCommunity(subtype=3, ipv4_address='192.0.2.1', local_administrator=65432), bgp.BGPOpaqueExtendedCommunity(subtype=13, opaque=b'abcdef'), bgp.BGPEncapsulationExtendedCommunity(subtype=12, tunnel_type=10), bgp.BGPEvpnMacMobilityExtendedCommunity(subtype=0, flags=255, sequence_number=287454020), bgp.BGPEvpnEsiLabelExtendedCommunity(subtype=1, flags=255, label=b'\xff\xff\xff'), bgp.BGPEvpnEsiLabelExtendedCommunity(subtype=1, flags=255, mpls_label=1048575), bgp.BGPEvpnEsiLabelExtendedCommunity(subtype=1, flags=255, vni=16777215), bgp.BGPEvpnEsImportRTExtendedCommunity(subtype=2, es_import='aa:bb:cc:dd:ee:ff'), bgp.BGPUnknownExtendedCommunity(type_=99, value=b'abcdefg')]
    path_attributes = [bgp.BGPPathAttributeOrigin(value=1), bgp.BGPPathAttributeAsPath(value=[[1000], {1001, 1002}, [1003, 1004]]), bgp.BGPPathAttributeNextHop(value='192.0.2.199'), bgp.BGPPathAttributeMultiExitDisc(value=2000000000), bgp.BGPPathAttributeLocalPref(value=1000000000), bgp.BGPPathAttributeAtomicAggregate(), bgp.BGPPathAttributeAggregator(as_number=40000, addr='192.0.2.99'), bgp.BGPPathAttributeCommunities(communities=communities), bgp.BGPPathAttributeExtendedCommunities(communities=ecommunities), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_NO_TUNNEL_INFORMATION_PRESENT, label=b'\xff\xff\xff'), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_NO_TUNNEL_INFORMATION_PRESENT, tunnel_id=None), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_INGRESS_REPLICATION, mpls_label=1048575, tunnel_id=bgp.PmsiTunnelIdIngressReplication(tunnel_endpoint_ip='1.1.1.1')), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_INGRESS_REPLICATION, vni=16777215, tunnel_id=bgp.PmsiTunnelIdIngressReplication(tunnel_endpoint_ip='aa:bb:cc::dd:ee:ff')), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=2, label=b'\xff\xff\xff', tunnel_id=bgp.PmsiTunnelIdUnknown(value=b'test')), bgp.BGPPathAttributeAs4Path(value=[[1000000], {1000001, 1002}, [1003, 1000004]]), bgp.BGPPathAttributeAs4Aggregator(as_number=100040000, addr='192.0.2.99'), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP, safi=safi.MPLS_VPN, next_hop='1.1.1.1', nlri=mp_nlri), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP, safi=safi.MPLS_LABEL, next_hop='1.1.1.1', nlri=mp_nlri2), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP6, safi=safi.MPLS_VPN, next_hop=['2001:db8::1'], nlri=mp_nlri_v6), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP6, safi=safi.MPLS_LABEL, next_hop=['2001:db8::1', 'fe80::1'], nlri=mp_nlri2_v6), bgp.BGPPathAttributeMpUnreachNLRI(afi=afi.IP, safi=safi.MPLS_VPN, withdrawn_routes=mp_nlri), bgp.BGPPathAttributeUnknown(flags=0, type_=100, value=300 * b'bar')]
    nlri = [bgp.BGPNLRI(length=24, addr='203.0.113.1'), bgp.BGPNLRI(length=16, addr='203.0.113.0')]
    msg1 = bgp.BGPUpdate(withdrawn_routes=withdrawn_routes, path_attributes=path_attributes, nlri=nlri)
    jsondict = msg1.to_jsondict()
    msg2 = bgp.BGPUpdate.from_jsondict(jsondict['BGPUpdate'])
    self.assertEqual(str(msg1), str(msg2))