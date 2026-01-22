def deferred_provider_lookup(self, api, method):
    """Create descriptor that performs lookup of api and method on demand.

        For specialized cases, such as the enforcer "get_member_from_driver"
        which needs to be effectively a "classmethod", this method returns
        a smart descriptor object that does the lookup at runtime instead of
        at import time.

        :param api: The api to use, e.g. "identity_api"
        :type api: str
        :param method: the method on the api to return
        :type method: str
        """

    class DeferredProviderLookup(object):

        def __init__(self, api, method):
            self.__api = api
            self.__method = method

        def __get__(self, instance, owner):
            api = getattr(ProviderAPIs, self.__api)
            return getattr(api, self.__method)
    return DeferredProviderLookup(api, method)