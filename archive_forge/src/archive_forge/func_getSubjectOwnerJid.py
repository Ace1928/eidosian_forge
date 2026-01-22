from .notification_groups import GroupsNotificationProtocolEntity
from yowsup.structs import ProtocolTreeNode
def getSubjectOwnerJid(self, full=True):
    return self.subjectOwnerJid if full else self.subjectOwnerJid.split('@')[0]