from parlai.core.worlds import World
from parlai.mturk.core.dev.agents import AssignState
def did_complete(self):
    """
        Determines whether or not this world was completed, or if the agent didn't
        complete the task.
        """
    agent = self.mturk_agent
    return not (agent.hit_is_abandoned or agent.hit_is_returned)