from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
import time

    <iq type="get" id="{{id}}" xmlns="urn:xmpp:whatsapp:sync">
        <sync
            sid="{{str((int(time.time()) + 11644477200) * 10000000)}}"
            index="{{0 | ?}}"
            last="{{true | false?}}"
        >
        </sync>
    </iq>
    