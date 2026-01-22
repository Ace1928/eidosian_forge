from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity

    When receiving a profile picture:
    <iq type="result" from="{{jid}}" id="{{id}}">
        <picture type="image" id="{{another_id}}">
        {{Binary bytes of the picture.}}
        </picture>
    </iq>
    