from yowsup.structs import ProtocolEntity, ProtocolTreeNode
from .iq_groups_participants  import ParticipantsGroupsIqProtocolEntity

    <iq type="set" id="{{id}}" xmlns="w:g2", to="{{group_jid}}">
        <promote>
            <participant jid="{{jid}}"></participant>
            <participant jid="{{jid}}"></participant>
        </promote>
    </iq>
    