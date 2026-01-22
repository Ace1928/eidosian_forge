from twisted.python import log
from twisted.words.xish import xpath

        Dispatch an event.

        When C{event} is L{None}, an XPath type event is triggered, and
        C{obj} is assumed to be an instance of
        L{Element<twisted.words.xish.domish.Element>}. Otherwise, C{event}
        holds the name of the named event being triggered. In the latter case,
        C{obj} can be anything.

        @param obj: The object to be dispatched.
        @param event: Optional event name.
        @type event: C{str}
        