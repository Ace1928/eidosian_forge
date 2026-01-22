from __future__ import annotations
import inspect
import uuid
from typing import (
from typing_extensions import TypedDict
from langchain_core._api import deprecated
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs
Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts a single example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool
        correctly.

    The ToolMessage is required because some chat models are hyper-optimized for agents
    rather than for an extraction use case.

    Arguments:
        input: string, the user input
        tool_calls: List[BaseModel], a list of tool calls represented as Pydantic
            BaseModels
        tool_outputs: Optional[List[str]], a list of tool call outputs.
            Does not need to be provided. If not provided, a placeholder value
            will be inserted.

    Returns:
        A list of messages

    Examples:

        .. code-block:: python

            from typing import List, Optional
            from langchain_core.pydantic_v1 import BaseModel, Field
            from langchain_openai import ChatOpenAI

            class Person(BaseModel):
                '''Information about a person.'''
                name: Optional[str] = Field(..., description="The name of the person")
                hair_color: Optional[str] = Field(
                    ..., description="The color of the peron's eyes if known"
                )
                height_in_meters: Optional[str] = Field(
                    ..., description="Height in METERs"
                )

            examples = [
                (
                    "The ocean is vast and blue. It's more than 20,000 feet deep.",
                    Person(name=None, height_in_meters=None, hair_color=None),
                ),
                (
                    "Fiona traveled far from France to Spain.",
                    Person(name="Fiona", height_in_meters=None, hair_color=None),
                ),
            ]


            messages = []

            for txt, tool_call in examples:
                messages.extend(
                    tool_example_to_messages(txt, [tool_call])
                )
    