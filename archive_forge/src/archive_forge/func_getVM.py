from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def getVM(self, name):
    self.__get_conn()
    VM = self.conn.get_VM(name)
    if VM:
        vminfo = dict()
        vminfo['uuid'] = VM.id
        vminfo['name'] = VM.name
        vminfo['status'] = VM.status.state
        vminfo['cpu_cores'] = VM.cpu.topology.cores
        vminfo['cpu_sockets'] = VM.cpu.topology.sockets
        vminfo['cpu_shares'] = VM.cpu_shares
        vminfo['memory'] = int(VM.memory) // 1024 // 1024 // 1024
        vminfo['mem_pol'] = int(VM.memory_policy.guaranteed) // 1024 // 1024 // 1024
        vminfo['os'] = VM.get_os().type_
        vminfo['del_prot'] = VM.delete_protected
        try:
            vminfo['host'] = str(self.conn.get_Host_byid(str(VM.host.id)).name)
        except Exception:
            vminfo['host'] = None
        vminfo['boot_order'] = []
        for boot_dev in VM.os.get_boot():
            vminfo['boot_order'].append(str(boot_dev.dev))
        vminfo['disks'] = []
        for DISK in VM.disks.list():
            disk = dict()
            disk['name'] = DISK.name
            disk['size'] = int(DISK.size) // 1024 // 1024 // 1024
            disk['domain'] = str(self.conn.get_domain_byid(DISK.get_storage_domains().get_storage_domain()[0].id).name)
            disk['interface'] = DISK.interface
            vminfo['disks'].append(disk)
        vminfo['ifaces'] = []
        for NIC in VM.nics.list():
            iface = dict()
            iface['name'] = str(NIC.name)
            iface['vlan'] = str(self.conn.get_network_byid(NIC.get_network().id).name)
            iface['interface'] = NIC.interface
            iface['mac'] = NIC.mac.address
            vminfo['ifaces'].append(iface)
            vminfo[str(NIC.name)] = NIC.mac.address
        CLUSTER = self.conn.get_cluster_byid(VM.cluster.id)
        if CLUSTER:
            vminfo['cluster'] = CLUSTER.name
    else:
        vminfo = False
    return vminfo