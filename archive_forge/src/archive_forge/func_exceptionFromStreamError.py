import copy
from typing import Optional
from twisted.words.xish import domish
def exceptionFromStreamError(element):
    """
    Build an exception object from a stream error.

    @param element: the stream error
    @type element: L{domish.Element}
    @return: the generated exception object
    @rtype: L{StreamError}
    """
    error = _parseError(element, NS_XMPP_STREAMS)
    exception = StreamError(error['condition'], error['text'], error['textLang'], error['appCondition'])
    return exception