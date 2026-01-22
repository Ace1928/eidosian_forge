from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import dpid as lib_dpid
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor_openflow
from os_ken.topology import event as topo_event
from os_ken.topology import api as topo_api
from . import vrrp_common

Usage:
osken-manager --verbose \
    os_ken.topology.switches \
    os_ken.tests.integrated.test_vrrp_multi \
    os_ken.services.protocols.vrrp.dumper

os_ken.services.protocols.vrrp.dumper is optional.

         +---+          ----------------
      /--|OVS|<--veth-->|              |
   OSKen   +---+          | linux bridge |<--veth--> command to generate packets
      \--|OVS|<--veth-->|              |
         +---+          ----------------

configure OVSs to connect os_ken
example
# ip link add br0 type bridge
# ip link add veth0-ovs type veth peer name veth0-br
# ip link add veth1-ovs type veth peer name veth1-br
# ip link set dev veth0-br master b0
# ip link set dev veth1-br master b0
# ip link show type bridge
22: b0: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether d6:97:42:8a:55:0e brd ff:ff:ff:ff:ff:ff

# bridge link show
23: veth0-br state DOWN @veth0-ovs: <BROADCAST,MULTICAST> mtu 1500 master b0 state disabled priority 32 cost 2
24: veth1-br state DOWN @veth1-ovs: <BROADCAST,MULTICAST> mtu 1500 master b0 state disabled priority 32 cost 2

# ovs-vsctl add-br s0
# ovs-vsctl add-port s0 veth0-ovs
# ovs-vsctl add-br s1
# ovs-vsctl add-port s1 veth1-ovs
# ovs-vsctl set-controller s0 tcp:127.0.0.1:6633
# ovs-vsctl set bridge s0 protocols='[OpenFlow12]'
# ovs-vsctl set-controller s1 tcp:127.0.0.1:6633
# ovs-vsctl set bridge s1 protocols='[OpenFlow12]'
# ovs-vsctl show
20c2a046-ae7e-4453-a576-11034db24985
    Manager "ptcp:6634"
    Bridge "s0"
        Controller "tcp:127.0.0.1:6633"
            is_connected: true
        Port "veth0-ovs"
            Interface "veth0-ovs"
        Port "s0"
            Interface "s0"
                type: internal
    Bridge "s1"
        Controller "tcp:127.0.0.1:6633"
            is_connected: true
        Port "veth1-ovs"
            Interface "veth1-ovs"
        Port "s1"
            Interface "s1"
                type: internal
    ovs_version: "1.9.90"
# ip link veth0-br set up
# ip link veth0-ovs set up
# ip link veth1-br set up
# ip link veth1-ovs set up
# ip link b0 set up
