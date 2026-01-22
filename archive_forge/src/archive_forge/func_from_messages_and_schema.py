from typing import (
from langchain_core._api.beta_decorator import beta
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts.chat import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import (
@classmethod
def from_messages_and_schema(cls, messages: Sequence[MessageLikeRepresentation], schema: Union[Dict, Type[BaseModel]]) -> ChatPromptTemplate:
    """Create a chat prompt template from a variety of message formats.

        Examples:

            Instantiation from a list of message templates:

            .. code-block:: python

                class OutputSchema(BaseModel):
                    name: str
                    value: int

                template = ChatPromptTemplate.from_messages(
                    [
                        ("human", "Hello, how are you?"),
                        ("ai", "I'm doing well, thanks!"),
                        ("human", "That's good to hear."),
                    ],
                    OutputSchema,
                )

        Args:
            messages: sequence of message representations.
                  A message can be represented using the following formats:
                  (1) BaseMessagePromptTemplate, (2) BaseMessage, (3) 2-tuple of
                  (message type, template); e.g., ("human", "{user_input}"),
                  (4) 2-tuple of (message class, template), (4) a string which is
                  shorthand for ("human", template); e.g., "{user_input}"
            schema: a dictionary representation of function call, or a Pydantic model.

        Returns:
            a structured prompt template
        """
    _messages = [_convert_to_message(message) for message in messages]
    input_vars: Set[str] = set()
    partial_vars: Dict[str, Any] = {}
    for _message in _messages:
        if isinstance(_message, MessagesPlaceholder) and _message.optional:
            partial_vars[_message.variable_name] = []
        elif isinstance(_message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)):
            input_vars.update(_message.input_variables)
    return cls(input_variables=sorted(input_vars), messages=_messages, partial_variables=partial_vars, schema_=schema)