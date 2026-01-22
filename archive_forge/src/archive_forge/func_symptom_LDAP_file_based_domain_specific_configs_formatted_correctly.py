import os
import re
import configparser
import keystone.conf
def symptom_LDAP_file_based_domain_specific_configs_formatted_correctly():
    """LDAP domain specific configuration files are not formatted correctly.

    If `keystone.conf [identity] domain_specific_drivers_enabled` is set
    to `true`, then support is enabled for individual domains to have their
    own identity drivers. The configurations for these can either be stored
    in a config file or in the database. The case we handle in this symptom
    is when they are stored in config files, which is indicated by
    `keystone.conf [identity] domain_configurations_from_database`
    being set to false. The config files located in the directory specified
    by `keystone.conf [identity] domain_config_dir` should be in the
    form of `keystone.<domain_name>.conf` and their contents should look
    something like this:

    [ldap]
    url = ldap://ldapservice.thecustomer.com
    query_scope = sub

    user_tree_dn = ou=Users,dc=openstack,dc=org
    user_objectclass = MyOrgPerson
    user_id_attribute = uid
    ...
    """
    filedir = CONF.identity.domain_config_dir
    if not CONF.identity.domain_specific_drivers_enabled or CONF.identity.domain_configurations_from_database or (not os.path.isdir(filedir)):
        return False
    invalid_files = []
    for filename in os.listdir(filedir):
        if re.match(CONFIG_REGEX, filename):
            try:
                parser = configparser.ConfigParser()
                parser.read(os.path.join(filedir, filename))
            except configparser.Error:
                invalid_files.append(filename)
    if invalid_files:
        invalid_str = ', '.join(invalid_files)
        print('Error: The following config files are formatted incorrectly: ', invalid_str)
        return True
    return False