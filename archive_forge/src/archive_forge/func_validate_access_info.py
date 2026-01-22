from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneclient import access
from keystoneclient import base
from keystoneclient.i18n import _
def validate_access_info(self, token):
    """Validate a token.

        :param token: Token to be validated. This can be an instance of
                      :py:class:`keystoneclient.access.AccessInfo` or a string
                      token_id.

        :rtype: :py:class:`keystoneclient.access.AccessInfoV2`

        """

    def calc_id(token):
        if isinstance(token, access.AccessInfo):
            return token.auth_token
        return base.getid(token)
    token_id = calc_id(token)
    body = self.get_token_data(token_id)
    return access.AccessInfo.factory(auth_token=token_id, body=body)