from .. import auth, utils
@utils.minimum_version('1.26')
@utils.check_resource('name')
def upgrade_plugin(self, name, remote, privileges):
    """
            Upgrade an installed plugin.

            Args:
                name (string): Name of the plugin to upgrade. The ``:latest``
                    tag is optional and is the default if omitted.
                remote (string): Remote reference to upgrade to. The
                    ``:latest`` tag is optional and is the default if omitted.
                privileges (:py:class:`list`): A list of privileges the user
                    consents to grant to the plugin. Can be retrieved using
                    :py:meth:`~plugin_privileges`.

            Returns:
                An iterable object streaming the decoded API logs
        """
    url = self._url('/plugins/{0}/upgrade', name)
    params = {'remote': remote}
    headers = {}
    registry, repo_name = auth.resolve_repository_name(remote)
    header = auth.get_config_header(self, registry)
    if header:
        headers['X-Registry-Auth'] = header
    response = self._post_json(url, params=params, headers=headers, data=privileges, stream=True)
    self._raise_for_status(response)
    return self._stream_helper(response, decode=True)