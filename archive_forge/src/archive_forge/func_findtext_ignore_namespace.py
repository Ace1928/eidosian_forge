def findtext_ignore_namespace(element, xpath, namespace=None, no_text_value=''):
    """
    Special version of findtext() which first tries to find the provided value using the provided
    namespace and in case no results are found we fallback to the xpath lookup without namespace.

    This is needed because some providers return some responses with namespace and some without.
    """
    result = findtext(element=element, xpath=xpath, namespace=namespace, no_text_value=no_text_value)
    if not result and namespace:
        result = findtext(element=element, xpath=xpath, namespace=None, no_text_value=no_text_value)
    return result