from zope.interface import Attribute, Interface
def getGroupConversation(group, Class, stayHidden=0):
    """
        For the given group object, returns the group conversation window or
        creates and returns a new group conversation window if it doesn't exist.

        @type group: L{Group<interfaces.IGroup>}
        @type Class: L{Conversation<interfaces.IConversation>} class
        @type stayHidden: boolean

        @rtype: L{GroupConversation<interfaces.IGroupConversation>}
        """