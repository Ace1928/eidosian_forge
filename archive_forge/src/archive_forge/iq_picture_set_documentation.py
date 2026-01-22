from .iq_picture import PictureIqProtocolEntity
from yowsup.structs import ProtocolTreeNode
import time

    <iq type="set" id="{{id}}" xmlns="w:profile:picture", to={{jid}}">
        <picture type="image" id="{{another_id}}">
        {{Binary bytes of the picture when type is set.}}
        </picture>
    </iq>
