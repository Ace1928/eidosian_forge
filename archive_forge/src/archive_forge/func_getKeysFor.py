from yowsup.layers import YowProtocolLayer
from yowsup.layers.axolotl.protocolentities import *
from yowsup.layers.network.layer import YowNetworkLayer
from yowsup.layers import EventCallback
from yowsup.profile.profile import YowProfile
from yowsup.axolotl import exceptions
from yowsup.layers.axolotl.props import PROP_IDENTITY_AUTOTRUST
import logging
def getKeysFor(self, jids, resultClbk, errorClbk=None, reason=None):
    logger.debug('getKeysFor(jids=%s, resultClbk=[omitted], errorClbk=[omitted], reason=%s)' % (jids, reason))

    def onSuccess(resultNode, getKeysEntity):
        entity = ResultGetKeysIqProtocolEntity.fromProtocolTreeNode(resultNode)
        resultJids = entity.getJids()
        successJids = []
        errorJids = entity.getErrors()
        for jid in getKeysEntity.jids:
            if jid not in resultJids:
                self.skipEncJids.append(jid)
                continue
            recipient_id = jid.split('@')[0]
            preKeyBundle = entity.getPreKeyBundleFor(jid)
            try:
                self.manager.create_session(recipient_id, preKeyBundle, autotrust=self.getProp(PROP_IDENTITY_AUTOTRUST, False))
                successJids.append(jid)
            except exceptions.UntrustedIdentityException as e:
                errorJids[jid] = e
                logger.error(e)
                logger.warning('Ignoring message with untrusted identity')
        resultClbk(successJids, errorJids)

    def onError(errorNode, getKeysEntity):
        if errorClbk:
            errorClbk(errorNode, getKeysEntity)
    entity = GetKeysIqProtocolEntity(jids, reason=reason)
    self._sendIq(entity, onSuccess, onError=onError)