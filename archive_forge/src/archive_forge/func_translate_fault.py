import logging
from oslo_vmware._i18n import _
def translate_fault(localized_method_fault, excep_msg=None):
    """Produce proper VimException subclass object,

    The exception is based on a vmodl.LocalizedMethodFault.

    :param excep_msg: Message to set to the exception. Defaults to
                      localizedMessage of the fault.
    """
    try:
        if not excep_msg:
            excep_msg = str(localized_method_fault.localizedMessage)
        name = localized_method_fault.fault.__class__.__name__
        fault_class = get_fault_class(name)
        if fault_class:
            ex = fault_class(excep_msg)
        else:
            ex = VimFaultException([name], excep_msg)
    except Exception as e:
        LOG.debug('Unexpected exception thrown (%s) while translating fault (%s) with message: %s.', e, localized_method_fault, excep_msg)
        ex = VimException(message=excep_msg, cause=e)
    return ex