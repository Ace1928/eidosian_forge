from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode
import hashlib
import base64
import os
from yowsup.common.tools import WATools

    <iq to="s.whatsapp.net" type="set" xmlns="w:m">
        <media hash="{{b64_hash}}" type="{{type}}" size="{{size_bytes}}" orighash={{b64_orighash?}}></media>
    </iq>
    