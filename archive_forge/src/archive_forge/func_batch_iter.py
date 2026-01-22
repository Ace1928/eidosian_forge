import json
import logging
import multiprocessing
import os
from collections import OrderedDict
from queue import Queue, PriorityQueue
from typing import List, Tuple, Any
import cv2
import numpy as np
from multiprocess.pool import Pool
from minerl.herobraine.hero.agent_handler import HandlerCollection, AgentHandler
from minerl.herobraine.hero.handlers import RewardHandler
def batch_iter(self, batch_size):
    """
        Returns a generator for iterating through batches of the dataset.
        :param batch_size:
        :param number_of_workers:
        :param worker_batch_size:
        :param size_to_dequeue:
        :return:
        """
    logger.info('Starting batch iterator on {}'.format(self.data_dir))
    data_list = self._get_all_valid_recordings(self.data_dir)
    load_data_func = self._get_load_data_func(self.data_queue, self.nsteps, self.worker_batch_size, self.mission_handlers, self.observables, self.actionables, self.gamma)
    map_promise = self.processing_pool.map_async(load_data_func, data_list)
    start = 0
    incr = 0
    while not map_promise.ready() or not self.data_queue.empty() or (not self.random_queue.empty()):
        while not self.data_queue.empty() and (not self.random_queue.full()):
            for ex in self.data_queue.get():
                if not self.random_queue.full():
                    r_num = np.random.rand(1)[0] * (1 - start) + start
                    self.random_queue.put((r_num, ex))
                    incr += 1
                else:
                    break
        if incr > self.size_to_dequeue:
            if self.random_queue.qsize() < batch_size:
                if map_promise.ready():
                    break
                else:
                    continue
            batch_with_incr = [self.random_queue.get() for _ in range(batch_size)]
            r1, batch = zip(*batch_with_incr)
            start = 0
            traj_obs, traj_acts, traj_handlers, traj_n_obs, discounted_rewards, elapsed = zip(*batch)
            observation_batch = [HandlerCollection({o: np.asarray(traj_ob[i]) for i, o in enumerate(self.observables)}) for traj_ob in traj_obs]
            action_batch = [HandlerCollection({a: np.asarray(traj_act[i]) for i, a in enumerate(self.actionables)}) for traj_act in traj_acts]
            mission_handler_batch = [HandlerCollection({m: np.asarray(traj_handler[i]) for i, m in enumerate(self.mission_handlers)}) for traj_handler in traj_handlers]
            next_observation_batch = [HandlerCollection({o: np.asarray(traj_n_ob[i]) for i, o in enumerate(self.observables)}) for traj_n_ob in traj_n_obs]
            yield (observation_batch, action_batch, mission_handler_batch, next_observation_batch, discounted_rewards, elapsed)
    try:
        map_promise.get()
    except RuntimeError as e:
        logger.error('Failure in data pipeline: {}'.format(e))
    logger.info('Epoch complete.')