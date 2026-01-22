from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def customize_vm(self, vm_obj):
    custom_spec_name = self.params.get('customization_spec')
    if custom_spec_name:
        cc_mgr = self.content.customizationSpecManager
        if cc_mgr.DoesCustomizationSpecExist(name=custom_spec_name):
            temp_spec = cc_mgr.GetCustomizationSpec(name=custom_spec_name)
            self.customspec = temp_spec.spec
            return
        self.module.fail_json(msg="Unable to find customization specification '%s' in given configuration." % custom_spec_name)
    adaptermaps = []
    for network in self.params['networks']:
        guest_map = vim.vm.customization.AdapterMapping()
        guest_map.adapter = vim.vm.customization.IPSettings()
        if 'ip' in network and 'netmask' in network:
            guest_map.adapter.ip = vim.vm.customization.FixedIp()
            guest_map.adapter.ip.ipAddress = str(network['ip'])
            guest_map.adapter.subnetMask = str(network['netmask'])
        elif 'type' in network and network['type'] == 'dhcp':
            guest_map.adapter.ip = vim.vm.customization.DhcpIpGenerator()
        if 'ipv6' in network and 'netmaskv6' in network:
            guest_map.adapter.ipV6Spec = vim.vm.customization.IPSettings.IpV6AddressSpec()
            guest_map.adapter.ipV6Spec.ip = [vim.vm.customization.FixedIpV6()]
            guest_map.adapter.ipV6Spec.ip[0].ipAddress = str(network['ipv6'])
            guest_map.adapter.ipV6Spec.ip[0].subnetMask = int(network['netmaskv6'])
        elif 'typev6' in network and network['typev6'] == 'dhcp':
            guest_map.adapter.ipV6Spec = vim.vm.customization.IPSettings.IpV6AddressSpec()
            guest_map.adapter.ipV6Spec.ip = [vim.vm.customization.DhcpIpV6Generator()]
        if 'gateway' in network:
            guest_map.adapter.gateway = network['gateway']
        if 'gatewayv6' in network:
            guest_map.adapter.ipV6Spec.gateway = network['gatewayv6']
        if 'domain' in network:
            guest_map.adapter.dnsDomain = network['domain']
        elif self.params['customization']['domain'] is not None:
            guest_map.adapter.dnsDomain = self.params['customization']['domain']
        if 'dns_servers' in network:
            guest_map.adapter.dnsServerList = network['dns_servers']
        elif self.params['customization']['dns_servers'] is not None:
            guest_map.adapter.dnsServerList = self.params['customization']['dns_servers']
        adaptermaps.append(guest_map)
    globalip = vim.vm.customization.GlobalIPSettings()
    if self.params['customization']['dns_servers'] is not None:
        globalip.dnsServerList = self.params['customization']['dns_servers']
    dns_suffixes = []
    dns_suffix = self.params['customization']['dns_suffix']
    if dns_suffix:
        if isinstance(dns_suffix, list):
            dns_suffixes += dns_suffix
        else:
            dns_suffixes.append(dns_suffix)
        globalip.dnsSuffixList = dns_suffixes
    if self.params['customization']['domain'] is not None:
        dns_suffixes.insert(0, self.params['customization']['domain'])
        globalip.dnsSuffixList = dns_suffixes
    if self.params['guest_id'] is not None:
        guest_id = self.params['guest_id']
    else:
        guest_id = vm_obj.summary.config.guestId
    if 'win' in guest_id:
        ident = vim.vm.customization.Sysprep()
        ident.userData = vim.vm.customization.UserData()
        ident.userData.computerName = vim.vm.customization.FixedName()
        default_name = ''
        if 'name' in self.params and self.params['name']:
            default_name = self.params['name'].replace(' ', '')
        elif vm_obj:
            default_name = vm_obj.name.replace(' ', '')
        punctuation = string.punctuation.replace('-', '')
        default_name = ''.join([c for c in default_name if c not in punctuation])
        if self.params['customization']['hostname'] is not None:
            ident.userData.computerName.name = self.params['customization']['hostname'][0:15]
        else:
            ident.userData.computerName.name = default_name[0:15]
        ident.userData.fullName = str(self.params['customization'].get('fullname', 'Administrator'))
        ident.userData.orgName = str(self.params['customization'].get('orgname', 'ACME'))
        if self.params['customization']['productid'] is not None:
            ident.userData.productId = str(self.params['customization']['productid'])
        ident.guiUnattended = vim.vm.customization.GuiUnattended()
        if self.params['customization']['autologon'] is not None:
            ident.guiUnattended.autoLogon = self.params['customization']['autologon']
            ident.guiUnattended.autoLogonCount = self.params['customization'].get('autologoncount', 1)
        if self.params['customization']['timezone'] is not None:
            ident.guiUnattended.timeZone = self.device_helper.integer_value(self.params['customization']['timezone'], 'customization.timezone')
        ident.identification = vim.vm.customization.Identification()
        if self.params['customization']['password'] is None or self.params['customization']['password'] == '':
            ident.guiUnattended.password = None
        else:
            ident.guiUnattended.password = vim.vm.customization.Password()
            ident.guiUnattended.password.value = str(self.params['customization']['password'])
            ident.guiUnattended.password.plainText = True
        if self.params['customization']['joindomain'] is not None:
            if self.params['customization']['domainadmin'] is None or self.params['customization']['domainadminpassword'] is None:
                self.module.fail_json(msg="'domainadmin' and 'domainadminpassword' entries are mandatory in 'customization' section to use joindomain feature")
            ident.identification.domainAdmin = self.params['customization']['domainadmin']
            ident.identification.joinDomain = self.params['customization']['joindomain']
            ident.identification.domainAdminPassword = vim.vm.customization.Password()
            ident.identification.domainAdminPassword.value = self.params['customization']['domainadminpassword']
            ident.identification.domainAdminPassword.plainText = True
        elif self.params['customization']['joinworkgroup'] is not None:
            ident.identification.joinWorkgroup = self.params['customization']['joinworkgroup']
        if self.params['customization']['runonce'] is not None:
            ident.guiRunOnce = vim.vm.customization.GuiRunOnce()
            ident.guiRunOnce.commandList = self.params['customization']['runonce']
    else:
        ident = vim.vm.customization.LinuxPrep()
        if self.params['customization']['domain'] is not None:
            ident.domain = self.params['customization']['domain']
        ident.hostName = vim.vm.customization.FixedName()
        default_name = ''
        if 'name' in self.params and self.params['name']:
            default_name = self.params['name']
        elif vm_obj:
            default_name = vm_obj.name
        if self.params['customization']['hostname'] is not None:
            hostname = self.params['customization']['hostname'].split('.')[0]
        else:
            hostname = default_name.split('.')[0]
        valid_hostname = re.sub('[^a-zA-Z0-9\\-]', '', hostname)
        ident.hostName.name = valid_hostname
        if self.params['customization']['timezone'] is not None:
            ident.timeZone = self.params['customization']['timezone']
        if self.params['customization']['hwclockUTC'] is not None:
            ident.hwClockUTC = self.params['customization']['hwclockUTC']
        if self.params['customization']['script_text'] is not None:
            ident.scriptText = self.params['customization']['script_text']
    self.customspec = vim.vm.customization.Specification()
    self.customspec.nicSettingMap = adaptermaps
    self.customspec.globalIPSettings = globalip
    self.customspec.identity = ident