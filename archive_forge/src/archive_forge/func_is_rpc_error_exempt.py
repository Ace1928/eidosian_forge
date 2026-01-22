from ncclient.transport.parser import DefaultXMLParser
import sys
def is_rpc_error_exempt(self, error_text):
    """
        Check whether an RPC error message is excempt, thus NOT causing an exception.

        On some devices the RPC operations may indicate an error response, even though
        the operation actually succeeded. This may be in cases where a warning would be
        more appropriate. In that case, the client may be better advised to simply
        ignore that error and not raise an exception.

        Note that there is also the "raise_mode", set on session and manager, which
        controls the exception-raising behaviour in case of returned errors. This error
        filter here is independent of that: No matter what the raise_mode says, if the
        error message matches one of the exempt errors returned here, an exception
        will not be raised.

        The exempt error messages are defined in the _EXEMPT_ERRORS field of the device
        handler object and can be overwritten by child classes.  Wild cards are
        possible: Start and/or end with a '*' to indicate that the text can appear at
        the start, the end or the middle of the error message to still match. All
        comparisons are case insensitive.

        Return True/False depending on found match.

        """
    if error_text is not None:
        error_text = error_text.lower().strip()
    else:
        error_text = 'no error given'
    for ex in self._exempt_errors_exact_match:
        if error_text == ex:
            return True
    for ex in self._exempt_errors_startwith_wildcard_match:
        if error_text.endswith(ex):
            return True
    for ex in self._exempt_errors_endwith_wildcard_match:
        if error_text.startswith(ex):
            return True
    for ex in self._exempt_errors_full_wildcard_match:
        if ex in error_text:
            return True
    return False