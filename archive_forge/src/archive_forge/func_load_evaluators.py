from typing import Any, Dict, List, Optional, Sequence, Type, Union
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.evaluation.agents.trajectory_eval_chain import TrajectoryEvalChain
from langchain.evaluation.comparison import PairwiseStringEvalChain
from langchain.evaluation.comparison.eval_chain import LabeledPairwiseStringEvalChain
from langchain.evaluation.criteria.eval_chain import (
from langchain.evaluation.embedding_distance.base import (
from langchain.evaluation.exact_match.base import ExactMatchStringEvaluator
from langchain.evaluation.parsing.base import (
from langchain.evaluation.parsing.json_distance import JsonEditDistanceEvaluator
from langchain.evaluation.parsing.json_schema import JsonSchemaEvaluator
from langchain.evaluation.qa import ContextQAEvalChain, CotQAEvalChain, QAEvalChain
from langchain.evaluation.regex_match.base import RegexMatchStringEvaluator
from langchain.evaluation.schema import EvaluatorType, LLMEvalChain, StringEvaluator
from langchain.evaluation.scoring.eval_chain import (
from langchain.evaluation.string_distance.base import (
def load_evaluators(evaluators: Sequence[EvaluatorType], *, llm: Optional[BaseLanguageModel]=None, config: Optional[dict]=None, **kwargs: Any) -> List[Union[Chain, StringEvaluator]]:
    """Load evaluators specified by a list of evaluator types.

    Parameters
    ----------
    evaluators : Sequence[EvaluatorType]
        The list of evaluator types to load.
    llm : BaseLanguageModel, optional
        The language model to use for evaluation, if none is provided, a default
        ChatOpenAI gpt-4 model will be used.
    config : dict, optional
        A dictionary mapping evaluator types to additional keyword arguments,
        by default None
    **kwargs : Any
        Additional keyword arguments to pass to all evaluators.

    Returns
    -------
    List[Chain]
        The loaded evaluators.

    Examples
    --------
    >>> from langchain.evaluation import load_evaluators, EvaluatorType
    >>> evaluators = [EvaluatorType.QA, EvaluatorType.CRITERIA]
    >>> loaded_evaluators = load_evaluators(evaluators, criteria="helpfulness")
    """
    loaded = []
    for evaluator in evaluators:
        _kwargs = config.get(evaluator, {}) if config else {}
        loaded.append(load_evaluator(evaluator, llm=llm, **{**kwargs, **_kwargs}))
    return loaded