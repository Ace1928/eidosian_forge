from castellan.common.credentials import keystone_password
from castellan.common.credentials import keystone_token
from castellan.common.credentials import password
from castellan.common.credentials import token
from castellan.common import exception
from oslo_config import cfg
from oslo_log import log as logging
def credential_factory(conf=None, context=None):
    """This function provides a factory for credentials.

    It is used to create an appropriare credential object
    from a passed configuration. This should be called before
    making any calls to a key manager.

    :param conf: Configuration file which this factory method uses
    to generate a credential object. Note: In the future it will
    become a required field.
    :param context: Context used for authentication. It can be used
    in conjunction with the configuration file. If no conf is passed,
    then the context object will be converted to a KeystoneToken and
    returned. If a conf is passed then only the 'token' is grabbed from
    the context for the authentication types that require a token.
    :returns: A credential object used for authenticating with the
    Castellan key manager. Type of credential returned depends on
    config and/or context passed.
    """
    if conf:
        conf.register_opts(credential_opts, group=OPT_GROUP)
        if conf.key_manager.auth_type == 'token':
            if conf.key_manager.token:
                auth_token = conf.key_manager.token
            elif context:
                auth_token = context.auth_token
            else:
                raise exception.InsufficientCredentialDataError()
            return token.Token(auth_token)
        elif conf.key_manager.auth_type == 'password':
            return password.Password(conf.key_manager.username, conf.key_manager.password)
        elif conf.key_manager.auth_type == 'keystone_password':
            return keystone_password.KeystonePassword(conf.key_manager.password, auth_url=conf.key_manager.auth_url, username=conf.key_manager.username, user_id=conf.key_manager.user_id, user_domain_id=conf.key_manager.user_domain_id, user_domain_name=conf.key_manager.user_domain_name, trust_id=conf.key_manager.trust_id, domain_id=conf.key_manager.domain_id, domain_name=conf.key_manager.domain_name, project_id=conf.key_manager.project_id, project_name=conf.key_manager.project_name, project_domain_id=conf.key_manager.project_domain_id, project_domain_name=conf.key_manager.project_domain_name, reauthenticate=conf.key_manager.reauthenticate)
        elif conf.key_manager.auth_type == 'keystone_token':
            if conf.key_manager.token:
                auth_token = conf.key_manager.token
            elif context:
                auth_token = context.auth_token
            else:
                raise exception.InsufficientCredentialDataError()
            return keystone_token.KeystoneToken(auth_token, auth_url=conf.key_manager.auth_url, trust_id=conf.key_manager.trust_id, domain_id=conf.key_manager.domain_id, domain_name=conf.key_manager.domain_name, project_id=conf.key_manager.project_id, project_name=conf.key_manager.project_name, project_domain_id=conf.key_manager.project_domain_id, project_domain_name=conf.key_manager.project_domain_name, reauthenticate=conf.key_manager.reauthenticate)
        else:
            LOG.error('Invalid auth_type specified.')
            raise exception.AuthTypeInvalidError(type=conf.key_manager.auth_type)
    if hasattr(context, 'tenant') and context.tenant:
        project_id = context.tenant
    elif hasattr(context, 'project_id') and context.project_id:
        project_id = context.project_id
    return keystone_token.KeystoneToken(context.auth_token, project_id=project_id)