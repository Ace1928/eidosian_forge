from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
def setContactStatus(self, person):
    """
        Inform the user that a person's status has changed.

        @param person: The person whose status has changed.
        @type person: L{IPerson<interfaces.IPerson>} provider
        """
    if person.name not in self.contacts:
        self.contacts[person.name] = person
    if person.name not in self.onlineContacts and (person.status == ONLINE or person.status == AWAY):
        self.onlineContacts[person.name] = person
    if person.name in self.onlineContacts and person.status == OFFLINE:
        del self.onlineContacts[person.name]