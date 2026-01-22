from __future__ import annotations
import asyncio
import logging
import time
from typing import (
from langchain_core.agents import (
from langchain_core.callbacks import (
from langchain_core.load.dump import dumpd
from langchain_core.outputs import RunInfo
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from langchain.schema import RUN_KEY
from langchain.utilities.asyncio import asyncio_timeout
def make_final_outputs(self, outputs: Dict[str, Any], run_manager: Union[CallbackManagerForChainRun, AsyncCallbackManagerForChainRun]) -> AddableDict:
    prepared_outputs = AddableDict(self.agent_executor.prep_outputs(self.inputs, outputs, return_only_outputs=True))
    if self.include_run_info:
        prepared_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
    return prepared_outputs