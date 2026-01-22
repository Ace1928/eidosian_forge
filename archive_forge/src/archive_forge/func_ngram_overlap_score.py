from typing import Dict, List
import numpy as np
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, root_validator
def ngram_overlap_score(source: List[str], example: List[str]) -> float:
    """Compute ngram overlap score of source and example as sentence_bleu score
    from NLTK package.

    Use sentence_bleu with method1 smoothing function and auto reweighting.
    Return float value between 0.0 and 1.0 inclusive.
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    hypotheses = source[0].split()
    references = [s.split() for s in example]
    return float(sentence_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1, auto_reweigh=True))