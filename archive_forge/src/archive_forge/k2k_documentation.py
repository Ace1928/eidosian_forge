from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import federation
from keystoneauth1 import plugin
Present ECP wrapped SAML assertion to the keystone SP.

        The assertion is issued by the keystone IdP and it is targeted to the
        keystone that will serve as Service Provider.

        :param session: a session object to send out HTTP requests.

        :param sp_url: URL where the ECP wrapped SAML assertion will be
                       presented to the keystone SP. Usually, something like:
                       https://sp.com/Shibboleth.sso/SAML2/ECP
        :type sp_url: str

        :param sp_auth_url: Federated authentication URL of the keystone SP.
                            It is specified by IdP, for example:
                            https://sp.com/v3/OS-FEDERATION/identity_providers/
                            idp_id/protocols/protocol_id/auth
        :type sp_auth_url: str

        