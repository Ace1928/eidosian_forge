from requests.auth import AuthBase, HTTPBasicAuth
from requests.compat import urlparse, urlunparse
def get_strategy_for(self, url):
    """Retrieve the authentication strategy for a specified URL.

        :param str url: The full URL you will be making a request against. For
            example, ``'https://api.github.com/user'``
        :returns: Callable that adds authentication to a request.

        .. code-block:: python

            import requests
            a = AuthHandler({'example.com', ('foo', 'bar')})
            strategy = a.get_strategy_for('http://example.com/example')
            assert isinstance(strategy, requests.auth.HTTPBasicAuth)

        """
    key = self._key_from_url(url)
    return self.strategies.get(key, NullAuthStrategy())