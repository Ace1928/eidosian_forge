from keystoneauth1.loading import base
from keystoneauth1.loading import opts
def register_conf_options(conf, group):
    """Register the oslo_config options that are needed for a plugin.

    This only registers the basic options shared by all plugins. Options that
    are specific to a plugin are loaded just before they are read.

    The defined options are:

     - auth_type: the name of the auth plugin that will be used for
         authentication.
     - auth_section: the group from which further auth plugin options should be
         taken. If section is not provided then the auth plugin options will be
         taken from the same group as provided in the parameters.

    :param conf: config object to register with.
    :type conf: oslo_config.cfg.ConfigOpts
    :param string group: The ini group to register options in.
    """
    conf.register_opt(_AUTH_SECTION_OPT._to_oslo_opt(), group=group)
    if conf[group].auth_section:
        group = conf[group].auth_section
    conf.register_opt(_AUTH_TYPE_OPT._to_oslo_opt(), group=group)