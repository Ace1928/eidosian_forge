from typing import Any, Mapping, Optional, Protocol
from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain.chains import ReduceDocumentsChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import (
from langchain.chains.question_answering.map_rerank_prompt import (
def load_qa_chain(llm: BaseLanguageModel, chain_type: str='stuff', verbose: Optional[bool]=None, callback_manager: Optional[BaseCallbackManager]=None, **kwargs: Any) -> BaseCombineDocumentsChain:
    """Load question answering chain.

    Args:
        llm: Language Model to use in the chain.
        chain_type: Type of document combining chain to use. Should be one of "stuff",
            "map_reduce", "map_rerank", and "refine".
        verbose: Whether chains should be run in verbose mode or not. Note that this
            applies to all chains that make up the final chain.
        callback_manager: Callback manager to use for the chain.

    Returns:
        A chain to use for question answering.
    """
    loader_mapping: Mapping[str, LoadingCallable] = {'stuff': _load_stuff_chain, 'map_reduce': _load_map_reduce_chain, 'refine': _load_refine_chain, 'map_rerank': _load_map_rerank_chain}
    if chain_type not in loader_mapping:
        raise ValueError(f'Got unsupported chain type: {chain_type}. Should be one of {loader_mapping.keys()}')
    return loader_mapping[chain_type](llm, verbose=verbose, callback_manager=callback_manager, **kwargs)