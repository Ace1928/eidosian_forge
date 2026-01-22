from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_acks.protocolentities import OutgoingAckProtocolEntity
import logging
def recvNotification(self, node):
    if node['type'] == 'picture':
        if node.getChild('set'):
            self.toUpper(SetPictureNotificationProtocolEntity.fromProtocolTreeNode(node))
        elif node.getChild('delete'):
            self.toUpper(DeletePictureNotificationProtocolEntity.fromProtocolTreeNode(node))
        else:
            self.raiseErrorForNode(node)
    elif node['type'] == 'status':
        self.toUpper(StatusNotificationProtocolEntity.fromProtocolTreeNode(node))
    elif node['type'] in ['contacts', 'subject', 'w:gp2']:
        pass
    else:
        logger.warning('Unsupported notification type: %s ' % node['type'])
        logger.debug('Unsupported notification node: %s' % node)
    ack = OutgoingAckProtocolEntity(node['id'], 'notification', node['type'], node['from'], participant=node['participant'])
    self.toLower(ack.toProtocolTreeNode())