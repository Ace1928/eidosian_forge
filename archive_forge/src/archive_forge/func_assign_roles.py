from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
@staticmethod
def assign_roles(agents):
    for a in agents:
        a.disp_id = 'Agent'