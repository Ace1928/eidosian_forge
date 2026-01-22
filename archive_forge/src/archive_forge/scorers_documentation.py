from typing import Any, Callable, Dict, Iterable, Optional
from spacy.scorer import PRFScore, ROCAUCScore
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import SimpleFrozenList
Returns PRF and ROC AUC scores for a doc-level attribute with a
    dict with scores for each label like Doc.cats. The reported overall
    score depends on the scorer settings.

    examples (Iterable[Example]): Examples to score
    attr (str): The attribute to score.
    getter (Callable[[Doc, str], Any]): Defaults to getattr. If provided,
        getter(doc, attr) should return the values for the individual doc.
    labels (Iterable[str]): The set of possible labels. Defaults to [].
    multi_label (bool): Whether the attribute allows multiple labels.
        Defaults to True. When set to False (exclusive labels), missing
        gold labels are interpreted as 0.0.
    positive_label (str): The positive label for a binary task with
        exclusive classes. Defaults to None.
    threshold (float): Cutoff to consider a prediction "positive". Defaults
        to 0.5 for multi-label, and 0.0 (i.e. whatever's highest scoring)
        otherwise.
    RETURNS (Dict[str, Any]): A dictionary containing the scores, with
        inapplicable scores as None:
        for all:
            attr_score (one of attr_micro_f / attr_macro_f / attr_macro_auc),
            attr_score_desc (text description of the overall score),
            attr_micro_p,
            attr_micro_r,
            attr_micro_f,
            attr_macro_p,
            attr_macro_r,
            attr_macro_f,
            attr_macro_auc,
            attr_f_per_type,
            attr_auc_per_type
    