from collections import defaultdict
from typing import (
import numpy as np
from .errors import Errors
from .morphology import Morphology
from .tokens import Doc, Span, Token
from .training import Example
from .util import SimpleFrozenList, get_lang_class
@staticmethod
def score_links(examples: Iterable[Example], *, negative_labels: Iterable[str], **cfg) -> Dict[str, Any]:
    """Returns PRF for predicted links on the entity level.
        To disentangle the performance of the NEL from the NER,
        this method only evaluates NEL links for entities that overlap
        between the gold reference and the predictions.

        examples (Iterable[Example]): Examples to score
        negative_labels (Iterable[str]): The string values that refer to no annotation (e.g. "NIL")
        RETURNS (Dict[str, Any]): A dictionary containing the scores.

        DOCS: https://spacy.io/api/scorer#score_links
        """
    f_per_type = {}
    for example in examples:
        gold_ent_by_offset = {}
        for gold_ent in example.reference.ents:
            gold_ent_by_offset[gold_ent.start_char, gold_ent.end_char] = gold_ent
        for pred_ent in example.predicted.ents:
            gold_span = gold_ent_by_offset.get((pred_ent.start_char, pred_ent.end_char), None)
            if gold_span is not None:
                label = gold_span.label_
                if label not in f_per_type:
                    f_per_type[label] = PRFScore()
                gold = gold_span.kb_id_
                if gold is not None:
                    pred = pred_ent.kb_id_
                    if gold in negative_labels and pred in negative_labels:
                        pass
                    elif gold == pred:
                        f_per_type[label].tp += 1
                    elif gold in negative_labels:
                        f_per_type[label].fp += 1
                    elif pred in negative_labels:
                        f_per_type[label].fn += 1
                    else:
                        f_per_type[label].fp += 1
                        f_per_type[label].fn += 1
    micro_prf = PRFScore()
    for label_prf in f_per_type.values():
        micro_prf.tp += label_prf.tp
        micro_prf.fn += label_prf.fn
        micro_prf.fp += label_prf.fp
    n_labels = len(f_per_type) + 1e-100
    macro_p = sum((prf.precision for prf in f_per_type.values())) / n_labels
    macro_r = sum((prf.recall for prf in f_per_type.values())) / n_labels
    macro_f = sum((prf.fscore for prf in f_per_type.values())) / n_labels
    results = {f'nel_score': micro_prf.fscore, f'nel_score_desc': 'micro F', f'nel_micro_p': micro_prf.precision, f'nel_micro_r': micro_prf.recall, f'nel_micro_f': micro_prf.fscore, f'nel_macro_p': macro_p, f'nel_macro_r': macro_r, f'nel_macro_f': macro_f, f'nel_f_per_type': {k: v.to_dict() for k, v in f_per_type.items()}}
    return results