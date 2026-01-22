import sys
def get_vm_type():
    if PydevdVmType.vm_type is None:
        setup_type()
    return PydevdVmType.vm_type