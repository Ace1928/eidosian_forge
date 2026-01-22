import inspect
from typing import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain.chains import LLMChain
from langchain.output_parsers.ernie_functions import (
from langchain.utils.ernie_functions import convert_pydantic_to_ernie_function
def create_ernie_fn_runnable(functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]], llm: Runnable, prompt: BasePromptTemplate, *, output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]]=None, **kwargs: Any) -> Runnable:
    '''Create a runnable sequence that uses Ernie functions.

    Args:
        functions: A sequence of either dictionaries, pydantic.BaseModels classes, or
            Python functions. If dictionaries are passed in, they are assumed to
            already be a valid Ernie functions. If only a single
            function is passed in, then it will be enforced that the model use that
            function. pydantic.BaseModels and Python functions should have docstrings
            describing what the function does. For best results, pydantic.BaseModels
            should have descriptions of the parameters and Python functions should have
            Google Python style args descriptions in the docstring. Additionally,
            Python functions should only use primitive types (str, int, float, bool) or
            pydantic.BaseModels for arguments.
        llm: Language model to use, assumed to support the Ernie function-calling API.
        prompt: BasePromptTemplate to pass to the model.
        output_parser: BaseLLMOutputParser to use for parsing model outputs. By default
            will be inferred from the function types. If pydantic.BaseModels are passed
            in, then the OutputParser will try to parse outputs using those. Otherwise
            model outputs will simply be parsed as JSON. If multiple functions are
            passed in and they are not pydantic.BaseModels, the chain output will
            include both the name of the function that was returned and the arguments
            to pass to the function.

    Returns:
        A runnable sequence that will pass in the given functions to the model when run.

    Example:
        .. code-block:: python

                from typing import Optional

                from langchain.chains.ernie_functions import create_ernie_fn_chain
                from langchain_community.chat_models import ErnieBotChat
                from langchain.prompts import ChatPromptTemplate
                from langchain.pydantic_v1 import BaseModel, Field


                class RecordPerson(BaseModel):
                    """Record some identifying information about a person."""

                    name: str = Field(..., description="The person's name")
                    age: int = Field(..., description="The person's age")
                    fav_food: Optional[str] = Field(None, description="The person's favorite food")


                class RecordDog(BaseModel):
                    """Record some identifying information about a dog."""

                    name: str = Field(..., description="The dog's name")
                    color: str = Field(..., description="The dog's color")
                    fav_food: Optional[str] = Field(None, description="The dog's favorite food")


                llm = ErnieBotChat(model_name="ERNIE-Bot-4")
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("user", "Make calls to the relevant function to record the entities in the following input: {input}"),
                        ("assistant", "OK!"),
                        ("user", "Tip: Make sure to answer in the correct format"),
                    ]
                )
                chain = create_ernie_fn_runnable([RecordPerson, RecordDog], llm, prompt)
                chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
                # -> RecordDog(name="Harry", color="brown", fav_food="chicken")
    '''
    if not functions:
        raise ValueError('Need to pass in at least one function. Received zero.')
    ernie_functions = [convert_to_ernie_function(f) for f in functions]
    llm_kwargs: Dict[str, Any] = {'functions': ernie_functions, **kwargs}
    if len(ernie_functions) == 1:
        llm_kwargs['function_call'] = {'name': ernie_functions[0]['name']}
    output_parser = output_parser or get_ernie_output_parser(functions)
    return prompt | llm.bind(**llm_kwargs) | output_parser