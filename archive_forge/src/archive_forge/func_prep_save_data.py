from parlai.core.worlds import World
from parlai.mturk.core.dev.agents import AssignState
def prep_save_data(self, workers):
    """
        This prepares data to be saved for later review, including chats from individual
        worker perspectives.
        """
    custom_data = self.get_custom_task_data()
    save_data = {'custom_data': custom_data, 'worker_data': {}}
    agent = self.mturk_agent
    save_data['worker_data'][agent.worker_id] = {'worker_id': agent.worker_id, 'agent_id': agent.id, 'assignment_id': agent.assignment_id, 'task_data': self.task_data, 'response': self.response, 'completed': self.episode_done()}
    return save_data