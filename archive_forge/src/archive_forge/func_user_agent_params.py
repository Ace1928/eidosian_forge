import base64
@property
def user_agent_params(self):
    """Any parameters necessary to identify this user agent.

        By default this is an empty dict (because authentication
        details don't contain any information about the application
        making the request), but when a resource is protected by
        OAuth, the OAuth consumer name is part of the user agent.
        """
    return {}