from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_CD(self, vmname, cd_drive):
    VM = self.get_VM(vmname)
    try:
        if str(VM.status.state) == 'down':
            cdrom = params.CdRom(file=cd_drive)
            VM.cdroms.add(cdrom)
            setMsg('Attached the image.')
            setChanged()
        else:
            cdrom = VM.cdroms.get(id='00000000-0000-0000-0000-000000000000')
            cdrom.set_file(cd_drive)
            cdrom.update(current=True)
            setMsg('Attached the image.')
            setChanged()
    except Exception as e:
        setMsg('Failed to attach image.')
        setMsg(str(e))
        setFailed()
        return False
    return True