from .cli import Cli, clicmd
from yowsup.layers.interface import YowInterfaceLayer, ProtocolEntityCallback
from yowsup.layers import YowLayerEvent, EventCallback
from yowsup.layers.network import YowNetworkLayer
import sys
from yowsup.common import YowConstants
import datetime
import time
import os
import logging
import threading
import base64
from yowsup.layers.protocol_groups.protocolentities      import *
from yowsup.layers.protocol_presence.protocolentities    import *
from yowsup.layers.protocol_messages.protocolentities    import *
from yowsup.layers.protocol_ib.protocolentities          import *
from yowsup.layers.protocol_iq.protocolentities          import *
from yowsup.layers.protocol_contacts.protocolentities    import *
from yowsup.layers.protocol_chatstate.protocolentities   import *
from yowsup.layers.protocol_privacy.protocolentities     import *
from yowsup.layers.protocol_media.protocolentities       import *
from yowsup.layers.protocol_media.mediauploader import MediaUploader
from yowsup.layers.protocol_profiles.protocolentities    import *
from yowsup.common.tools import Jid
from yowsup.common.optionalmodules import PILOptionalModule
from yowsup.layers.axolotl.protocolentities.iq_key_get import GetKeysIqProtocolEntity
@clicmd('Get lastseen for contact')
def contact_lastseen(self, jid):
    if self.assertConnected():

        def onSuccess(resultIqEntity, originalIqEntity):
            self.output('%s lastseen %s seconds ago' % (resultIqEntity.getFrom(), resultIqEntity.getSeconds()))

        def onError(errorIqEntity, originalIqEntity):
            logger.error('Error getting lastseen information for %s' % originalIqEntity.getTo())
        entity = LastseenIqProtocolEntity(self.aliasToJid(jid))
        self._sendIq(entity, onSuccess, onError)