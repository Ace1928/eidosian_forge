from yowsup.structs import ProtocolEntity, ProtocolTreeNode
from .iq_groups import GroupsIqProtocolEntity

    <iq type="set" id="{{id}}" xmlns="w:g2", to={{group_jid}}">
        <subject>
              {{NEW_VAL}}
        </subject>
    </iq>
    