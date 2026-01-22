import keystone.conf
def symptom_debug_mode_is_enabled():
    """Debug mode should be set to False.

    Debug mode can be used to get more information back when trying to isolate
    a problem, but it is not recommended to be enabled when running a
    production environment.

    Ensure `keystone.conf debug` is set to False
    """
    return CONF.debug