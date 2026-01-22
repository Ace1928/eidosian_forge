from sys import intern
from typing import Type
from twisted.internet import protocol
from twisted.python import failure
from twisted.words.xish import domish, utility
def installBootstraps(self, dispatcher):
    """
        Install registered bootstrap observers.

        @param dispatcher: Event dispatcher to add the observers to.
        @type dispatcher: L{utility.EventDispatcher}
        """
    for event, fn in self.bootstraps:
        dispatcher.addObserver(event, fn)