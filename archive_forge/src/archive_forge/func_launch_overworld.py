import logging
import time
import datetime
from concurrent import futures
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
def launch_overworld(self, task_name, overworld_name, onboard_map, overworld_agent):
    """
        Launch an overworld and a subsequent onboarding world.

        Return the job's future

        :param task_name:
            string. the name of the job thread
        :param overworld_name:
            string. the name of the overworld in the module file
        :param onboard_map:
            map. a mapping of overworld return values to the names
            of onboarding worlds in the module file.
        :param overworld_agent:
            The agent to run the overworld with

        :return:
            the Futures object corresponding to running the overworld
        """
    task = utils.TaskState(task_name, overworld_name, [overworld_agent], is_overworld=True, world_type=None)
    self.tasks[task_name] = task
    agent_state = self.manager.get_agent_state(overworld_agent.id)

    def _world_function():
        world_generator = utils.get_world_fn_attr(self._world_module, overworld_name, 'generate_world')
        overworld = world_generator(self.opt, [overworld_agent])
        while not overworld.episode_done() and (not self.system_done):
            world_type = overworld.parley()
            if world_type is None:
                time.sleep(0.5)
                continue
            if world_type == self.manager.EXIT_STR:
                self.manager._remove_agent(overworld_agent.id)
                return world_type
            onboard_type = onboard_map.get(world_type)
            if onboard_type:
                onboard_id = 'onboard-{}-{}'.format(overworld_agent.id, time.time())
                agent = self.manager._create_agent(onboard_id, overworld_agent.id)
                agent.data = overworld_agent.data
                agent_state.set_active_agent(agent)
                agent_state.assign_agent_to_task(agent, onboard_id)
                _, onboard_data = self._run_world(task, onboard_type, [agent])
                agent_state.onboard_data = onboard_data
                agent_state.data = agent.data
            self.manager.add_agent_to_pool(agent_state, world_type)
            log_utils.print_and_log(logging.INFO, 'onboarding/overworld complete')
        return world_type
    fut = self.executor.submit(_world_function)
    task.future = fut
    return fut