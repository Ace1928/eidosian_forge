from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import find_obj, vmware_argument_spec, PyVmomi
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def get_config_option_recommended(self, guest_os_desc, hwv_version=''):
    guest_os_option_dict = {}
    support_usb_controller = []
    support_disk_controller = []
    support_ethernet_card = []
    if guest_os_desc and len(guest_os_desc) != 0:
        default_disk_ctl = default_ethernet = default_cdrom_ctl = default_usb_ctl = ''
        for name, dev_type in self.ctl_device_type.items():
            for supported_type in guest_os_desc[0].supportedUSBControllerList:
                if supported_type == dev_type:
                    support_usb_controller = support_usb_controller + [name]
                if dev_type == guest_os_desc[0].recommendedUSBController:
                    default_usb_ctl = name
            for supported_type in guest_os_desc[0].supportedEthernetCard:
                if supported_type == dev_type:
                    support_ethernet_card = support_ethernet_card + [name]
                if dev_type == guest_os_desc[0].recommendedEthernetCard:
                    default_ethernet = name
            for supported_type in guest_os_desc[0].supportedDiskControllerList:
                if supported_type == dev_type:
                    support_disk_controller = support_disk_controller + [name]
                if dev_type == guest_os_desc[0].recommendedDiskController:
                    default_disk_ctl = name
                if dev_type == guest_os_desc[0].recommendedCdromController:
                    default_cdrom_ctl = name
        guest_os_option_dict = {'hardware_version': hwv_version, 'guest_id': guest_os_desc[0].id, 'guest_fullname': guest_os_desc[0].fullName, 'rec_cpu_cores_per_socket': guest_os_desc[0].numRecommendedCoresPerSocket, 'rec_cpu_socket': guest_os_desc[0].numRecommendedPhysicalSockets, 'rec_memory_mb': guest_os_desc[0].recommendedMemMB, 'rec_firmware': guest_os_desc[0].recommendedFirmware, 'default_secure_boot': guest_os_desc[0].defaultSecureBoot, 'support_secure_boot': guest_os_desc[0].supportsSecureBoot, 'default_disk_controller': default_disk_ctl, 'rec_disk_mb': guest_os_desc[0].recommendedDiskSizeMB, 'default_ethernet': default_ethernet, 'default_cdrom_controller': default_cdrom_ctl, 'default_usb_controller': default_usb_ctl, 'support_tpm_20': guest_os_desc[0].supportsTPM20, 'support_persistent_memory': guest_os_desc[0].persistentMemorySupported, 'rec_persistent_memory': guest_os_desc[0].recommendedPersistentMemoryMB, 'support_min_persistent_mem_mb': guest_os_desc[0].supportedMinPersistentMemoryMB, 'rec_vram_kb': guest_os_desc[0].vRAMSizeInKB.defaultValue, 'support_usb_controller': support_usb_controller, 'support_disk_controller': support_disk_controller, 'support_ethernet_card': support_ethernet_card}
    return guest_os_option_dict