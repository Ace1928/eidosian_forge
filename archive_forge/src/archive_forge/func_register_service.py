import inspect
import logging
import sys
def register_service(service):
    """
    Register the os_ken application specified by 'service' as
    a provider of events defined in the calling module.

    If an application being loaded consumes events (in the sense of
    set_ev_cls) provided by the 'service' application, the latter
    application will be automatically loaded.

    This mechanism is used to e.g. automatically start ofp_handler if
    there are applications consuming OFP events.
    """
    frame = inspect.currentframe()
    if frame is not None:
        m_name = frame.f_back.f_globals['__name__']
        m = sys.modules[m_name]
        m._SERVICE_NAME = service