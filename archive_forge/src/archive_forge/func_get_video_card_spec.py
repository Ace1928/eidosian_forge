from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
def get_video_card_spec(self, vm_obj):
    """
        Get device changes of virtual machine
        Args:
            vm_obj: Managed object of virtual machine
        Returns: virtual device spec
        """
    video_card, video_card_facts = self.gather_video_card_facts(vm_obj)
    self.video_card_facts = video_card_facts
    if video_card is None:
        self.module.fail_json(msg='Unable to get video card device for the specified virtual machine.')
    video_spec = vim.vm.device.VirtualDeviceSpec()
    video_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
    video_spec.device = video_card
    auto_detect = False
    enabled_3d = False
    if self.params['gather_video_facts']:
        return None
    if self.params['use_auto_detect'] is not None:
        if video_card_facts['auto_detect'] and self.params['use_auto_detect']:
            auto_detect = True
        elif not video_card_facts['auto_detect'] and self.params['use_auto_detect']:
            video_spec.device.useAutoDetect = True
            self.change_detected = True
            auto_detect = True
        elif video_card_facts['auto_detect'] and (not self.params['use_auto_detect']):
            video_spec.device.useAutoDetect = False
            self.change_detected = True
    elif video_card_facts['auto_detect']:
        auto_detect = True
    if not auto_detect:
        if self.params['display_number'] is not None:
            if self.params['display_number'] < 1:
                self.module.fail_json(msg='display_number attribute valid value: 1-10.')
            if self.params['display_number'] != video_card_facts['display_number']:
                video_spec.device.numDisplays = self.params['display_number']
                self.change_detected = True
        if self.params['video_memory_mb'] is not None:
            if self.params['video_memory_mb'] < 1.172:
                self.module.fail_json(msg='video_memory_mb attribute valid value: ESXi 6.7U1(1.172-256 MB),ESXi 6.7/6.5/6.0(1.172-128 MB).')
            if int(self.params['video_memory_mb'] * 1024) != video_card_facts['video_memory']:
                video_spec.device.videoRamSizeInKB = int(self.params['video_memory_mb'] * 1024)
                self.change_detected = True
    elif self.params['display_number'] is not None or self.params['video_memory_mb'] is not None:
        self.module.fail_json(msg='display_number and video_memory_mb can not be changed if use_auto_detect is true.')
    if self.params['enable_3D'] is not None:
        if self.params['enable_3D'] != video_card_facts['enable_3D_support']:
            video_spec.device.enable3DSupport = self.params['enable_3D']
            self.change_detected = True
            if self.params['enable_3D']:
                enabled_3d = True
        elif video_card_facts['enable_3D_support']:
            enabled_3d = True
    elif video_card_facts['enable_3D_support']:
        enabled_3d = True
    if enabled_3d:
        if self.params['renderer_3D'] is not None:
            renderer = self.params['renderer_3D'].lower()
            if renderer not in ['automatic', 'software', 'hardware']:
                self.module.fail_json(msg='renderer_3D attribute valid value: automatic, software, hardware.')
            if renderer != video_card_facts['renderer_3D']:
                video_spec.device.use3dRenderer = renderer
                self.change_detected = True
        if self.params['memory_3D_mb'] is not None:
            memory_3d = self.params['memory_3D_mb']
            if not self.is_power_of_2(memory_3d):
                self.module.fail_json(msg='memory_3D_mb attribute should be an integer value and power of 2(32-2048).')
            elif memory_3d < 32 or memory_3d > 2048:
                self.module.fail_json(msg='memory_3D_mb attribute should be an integer value and power of 2(32-2048).')
            if memory_3d * 1024 != video_card_facts['memory_3D']:
                video_spec.device.graphicsMemorySizeInKB = memory_3d * 1024
                self.change_detected = True
    elif self.params['renderer_3D'] is not None or self.params['memory_3D_mb'] is not None:
        self.module.fail_json(msg='3D renderer or 3D memory can not be configured if 3D is not enabled.')
    if not self.change_detected:
        return None
    return video_spec