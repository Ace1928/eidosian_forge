from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool

    Example:

    ```py
    from transformers.tools import TextSummarizationTool

    summarizer = TextSummarizationTool()
    summarizer(long_text)
    ```
    