from packaging import version
import wandb.util
from wandb.sdk.lib import deprecate
from langchain.callbacks.tracers import WandbTracer  # noqa: E402, I001
This module contains an integration with the LangChain library.

Specifically, it exposes a `WandbTracer` class that can be used to stream
LangChain activity to W&B. The intended usage pattern is to call
`tracer = WandbTracer()` at the top of the script/notebook, and call
`tracer.finish()` at the end of the script/notebook.
 This will stream all LangChain activity to W&B.

Technical Note:
LangChain is in very rapid development - meaning their APIs and schemas are actively changing.
As a matter of precaution, any call to LangChain apis, or use of their returned data is wrapped
in a try/except block. This is to ensure that if a breaking change is introduced, the W&B
integration will not break user code. The one exception to the rule is at import time. If
LangChain is not installed, or the symbols are not in the same place, the appropriate error
will be raised when importing this module.
