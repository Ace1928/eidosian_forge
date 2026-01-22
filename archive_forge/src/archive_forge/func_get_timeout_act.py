import time
from typing import List, Optional
from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.worlds import World
@staticmethod
def get_timeout_act(agent: Agent, timeout: int=DEFAULT_TIMEOUT, quick_replies: Optional[List[str]]=None) -> Optional[Message]:
    """
        Return an agent's act, with a specified timeout.

        :param agent:
            Agent who is acting
        :param timeout:
            how long to wait
        :param quick_replies:
            If given, agent's message *MUST* be one of the quick replies

        :return:
            An act dictionary if no timeout; else, None
        """

    def _is_valid(act):
        return act.get('text', '') in quick_replies if quick_replies else True
    act = None
    curr_time = time.time()
    allowed_timeout = timeout
    while act is None and time.time() - curr_time < allowed_timeout:
        act = agent.act()
        if act is not None and (not _is_valid(act)):
            agent.observe({'id': '', 'text': 'Invalid response. Please choose one of the quick replies', 'quick_replies': quick_replies})
        time.sleep(THREAD_SLEEP)
    return act