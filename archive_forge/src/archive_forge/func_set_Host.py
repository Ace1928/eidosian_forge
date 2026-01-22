from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_Host(self, host_name, cluster, ifaces):
    HOST = self.get_Host(host_name)
    CLUSTER = self.get_cluster(cluster)
    if HOST is None:
        setMsg('Host does not exist.')
        ifacelist = dict()
        networklist = []
        manageip = ''
        try:
            for iface in ifaces:
                try:
                    setMsg('creating host interface ' + iface['name'])
                    if 'management' in iface:
                        manageip = iface['ip']
                    if 'boot_protocol' not in iface:
                        if 'ip' in iface:
                            iface['boot_protocol'] = 'static'
                        else:
                            iface['boot_protocol'] = 'none'
                    if 'ip' not in iface:
                        iface['ip'] = ''
                    if 'netmask' not in iface:
                        iface['netmask'] = ''
                    if 'gateway' not in iface:
                        iface['gateway'] = ''
                    if 'network' in iface:
                        if 'bond' in iface:
                            bond = []
                            for slave in iface['bond']:
                                bond.append(ifacelist[slave])
                            try:
                                tmpiface = params.Bonding(slaves=params.Slaves(host_nic=bond), options=params.Options(option=[params.Option(name='miimon', value='100'), params.Option(name='mode', value='4')]))
                            except Exception as e:
                                setMsg('Failed to create the bond for  ' + iface['name'])
                                setFailed()
                                setMsg(str(e))
                                return False
                            try:
                                tmpnetwork = params.HostNIC(network=params.Network(name=iface['network']), name=iface['name'], boot_protocol=iface['boot_protocol'], ip=params.IP(address=iface['ip'], netmask=iface['netmask'], gateway=iface['gateway']), override_configuration=True, bonding=tmpiface)
                                networklist.append(tmpnetwork)
                                setMsg('Applying network ' + iface['name'])
                            except Exception as e:
                                setMsg('Failed to set' + iface['name'] + ' as network interface')
                                setFailed()
                                setMsg(str(e))
                                return False
                        else:
                            tmpnetwork = params.HostNIC(network=params.Network(name=iface['network']), name=iface['name'], boot_protocol=iface['boot_protocol'], ip=params.IP(address=iface['ip'], netmask=iface['netmask'], gateway=iface['gateway']))
                            networklist.append(tmpnetwork)
                            setMsg('Applying network ' + iface['name'])
                    else:
                        tmpiface = params.HostNIC(name=iface['name'], network=params.Network(), boot_protocol=iface['boot_protocol'], ip=params.IP(address=iface['ip'], netmask=iface['netmask'], gateway=iface['gateway']))
                    ifacelist[iface['name']] = tmpiface
                except Exception as e:
                    setMsg('Failed to set ' + iface['name'])
                    setFailed()
                    setMsg(str(e))
                    return False
        except Exception as e:
            setMsg('Failed to set networks')
            setMsg(str(e))
            setFailed()
            return False
        if manageip == '':
            setMsg('No management network is defined')
            setFailed()
            return False
        try:
            HOST = params.Host(name=host_name, address=manageip, cluster=CLUSTER, ssh=params.SSH(authentication_method='publickey'))
            if self.conn.hosts.add(HOST):
                setChanged()
                HOST = self.get_Host(host_name)
                state = HOST.status.state
                while state != 'non_operational' and state != 'up':
                    HOST = self.get_Host(host_name)
                    state = HOST.status.state
                    time.sleep(1)
                    if state == 'non_responsive':
                        setMsg('Failed to add host to RHEVM')
                        setFailed()
                        return False
                setMsg('status host: up')
                time.sleep(5)
                HOST = self.get_Host(host_name)
                state = HOST.status.state
                setMsg('State before setting to maintenance: ' + str(state))
                HOST.deactivate()
                while state != 'maintenance':
                    HOST = self.get_Host(host_name)
                    state = HOST.status.state
                    time.sleep(1)
                setMsg('status host: maintenance')
                try:
                    HOST.nics.setupnetworks(params.Action(force=True, check_connectivity=False, host_nics=params.HostNics(host_nic=networklist)))
                    setMsg('nics are set')
                except Exception as e:
                    setMsg('Failed to apply networkconfig')
                    setFailed()
                    setMsg(str(e))
                    return False
                try:
                    HOST.commitnetconfig()
                    setMsg('Network config is saved')
                except Exception as e:
                    setMsg('Failed to save networkconfig')
                    setFailed()
                    setMsg(str(e))
                    return False
        except Exception as e:
            if 'The Host name is already in use' in str(e):
                setMsg('Host already exists')
            else:
                setMsg('Failed to add host')
                setFailed()
                setMsg(str(e))
            return False
        HOST.activate()
        while state != 'up':
            HOST = self.get_Host(host_name)
            state = HOST.status.state
            time.sleep(1)
            if state == 'non_responsive':
                setMsg('Failed to apply networkconfig.')
                setFailed()
                return False
        setMsg('status host: up')
    else:
        setMsg('Host exists.')
    return True