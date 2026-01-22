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
def load_evaluator(evaluator: EvaluatorType, *, llm: Optional[BaseLanguageModel]=None, **kwargs: Any) -> Union[Chain, StringEvaluator]:
    """Load the requested evaluation chain specified by a string.

    Parameters
    ----------
    evaluator : EvaluatorType
        The type of evaluator to load.
    llm : BaseLanguageModel, optional
        The language model to use for evaluation, by default None
    **kwargs : Any
        Additional keyword arguments to pass to the evaluator.

    Returns
    -------
    Chain
        The loaded evaluation chain.

    Examples
    --------
    >>> from langchain.evaluation import load_evaluator, EvaluatorType
    >>> evaluator = load_evaluator(EvaluatorType.QA)
    """
    if evaluator not in _EVALUATOR_MAP:
        raise ValueError(f'Unknown evaluator type: {evaluator}\nValid types are: {list(_EVALUATOR_MAP.keys())}')
    evaluator_cls = _EVALUATOR_MAP[evaluator]
    if issubclass(evaluator_cls, LLMEvalChain):
        try:
            llm = llm or ChatOpenAI(model='gpt-4', model_kwargs={'seed': 42}, temperature=0)
        except Exception as e:
            raise ValueError(f"Evaluation with the {evaluator_cls} requires a language model to function. Failed to create the default 'gpt-4' model. Please manually provide an evaluation LLM or check your openai credentials.") from e
        return evaluator_cls.from_llm(llm=llm, **kwargs)
    else:
        return evaluator_cls(**kwargs)