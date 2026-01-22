from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence, Tuple, Union
from warnings import warn
from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.config import run_in_executor
from langchain.chains.base import Chain
def evaluate_agent_trajectory(self, *, prediction: str, agent_trajectory: Sequence[Tuple[AgentAction, str]], input: str, reference: Optional[str]=None, **kwargs: Any) -> dict:
    """Evaluate a trajectory.

        Args:
            prediction (str): The final predicted response.
            agent_trajectory (List[Tuple[AgentAction, str]]):
                The intermediate steps forming the agent trajectory.
            input (str): The input to the agent.
            reference (Optional[str]): The reference answer.

        Returns:
            dict: The evaluation result.
        """
    self._check_evaluation_args(reference=reference, input=input)
    return self._evaluate_agent_trajectory(prediction=prediction, input=input, agent_trajectory=agent_trajectory, reference=reference, **kwargs)