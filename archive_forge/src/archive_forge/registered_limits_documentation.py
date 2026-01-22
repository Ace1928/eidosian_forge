from keystoneclient import base
Delete a registered limit.

        :param registered_limit: the registered limit to delete.
        :type registered_limit:
            str or :class:`keystoneclient.v3.registered_limits.RegisteredLimit`

        :returns: Response object with 204 status.
        :rtype: :class:`requests.models.Response`

        