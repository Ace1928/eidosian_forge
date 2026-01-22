from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def license_probe(self):
    props = []
    cmd = 'lslicense'
    cmdopts = {}
    data = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    if self.remote and int(data['license_remote']) != self.remote:
        props += ['remote']
    if self.virtualization and int(data['license_virtualization']) != self.virtualization:
        props += ['virtualization']
    if self.compression:
        if self.system_data['product_name'] == 'IBM Storwize V7000' or self.system_data['product_name'] == 'IBM FlashSystem 7200':
            if int(data['license_compression_enclosures']) != self.compression:
                self.log('license_compression_enclosure=%d', int(data['license_compression_enclosures']))
                props += ['compression']
        elif int(data['license_compression_capacity']) != self.compression:
            self.log('license_compression_capacity=%d', int(data['license_compression_capacity']))
            props += ['compression']
    if self.flash and int(data['license_flash']) != self.flash:
        props += ['flash']
    if self.cloud and int(data['license_cloud_enclosures']) != self.cloud:
        props += ['cloud']
    if self.easytier and int(data['license_easy_tier']) != self.easytier:
        props += ['easytier']
    if self.physical_flash and data['license_physical_flash'] != self.physical_flash:
        props += ['physical_flash']
    self.log('props: %s', props)
    return props