from .iq_picture import PictureIqProtocolEntity
from yowsup.structs import ProtocolTreeNode

    <iq type="result" from="{{jid}}" id="{{id}}">
        <picture type="image | preview" id="{{another_id}}">
        {{Binary bytes of the picture.}}
        </picture>
    </iq>
    