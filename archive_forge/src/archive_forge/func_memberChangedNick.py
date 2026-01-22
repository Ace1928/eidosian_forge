from zope.interface import Attribute, Interface
def memberChangedNick(oldnick, newnick):
    """
        Changes the oldnick in the list of members to C{newnick} and displays this
        change to the user,

        @type oldnick: string (XXX: Not Person?)
        @type newnick: string
        """