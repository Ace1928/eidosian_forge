from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
def remote_setGroupMetadata(self, dict_, groupName):
    if 'topic' in dict_:
        self.getGroupConversation(groupName).setTopic(dict_['topic'], dict_.get('topic_author', None))