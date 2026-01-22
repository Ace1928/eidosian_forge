from typing import Any, Dict, Optional, TextIO, cast
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils.input import print_text
def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
    """Print out that we are entering a chain."""
    class_name = serialized.get('name', serialized.get('id', ['<unknown>'])[-1])
    print_text(f'\n\n\x1b[1m> Entering new {class_name} chain...\x1b[0m', end='\n', file=self.file)