from typing import Any, Dict, Optional, TextIO, cast
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils.input import print_text
def on_agent_finish(self, finish: AgentFinish, color: Optional[str]=None, **kwargs: Any) -> None:
    """Run on agent end."""
    print_text(finish.log, color=color or self.color, end='\n', file=self.file)