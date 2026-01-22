def get_ca_certificate(self):
    """Get CA certificate.

        :returns: PEM-formatted string.
        :rtype: str

        """
    resp, body = self._client.get('/certificates/ca', authenticated=False)
    return resp.text