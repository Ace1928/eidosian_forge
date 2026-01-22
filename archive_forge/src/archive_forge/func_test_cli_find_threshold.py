import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest
import srsly
from click import NoSuchOption
from packaging.specifiers import SpecifierSet
from thinc.api import Config
import spacy
from spacy import about
from spacy.cli import info
from spacy.cli._util import parse_config_overrides, string_to_list, walk_directory
from spacy.cli.apply import apply
from spacy.cli.debug_data import (
from spacy.cli.download import get_compatibility, get_version
from spacy.cli.evaluate import render_parses
from spacy.cli.find_threshold import find_threshold
from spacy.cli.init_config import RECOMMENDATIONS, fill_config, init_config
from spacy.cli.init_pipeline import _init_labels
from spacy.cli.package import _is_permitted_package_name, get_third_party_dependencies
from spacy.cli.validate import get_model_pkgs
from spacy.lang.en import English
from spacy.lang.nl import Dutch
from spacy.language import Language
from spacy.schemas import RecommendationSchema
from spacy.tokens import Doc, DocBin
from spacy.tokens.span import Span
from spacy.training import Example, docs_to_json, offsets_to_biluo_tags
from spacy.training.converters import conll_ner_to_docs, conllu_to_docs, iob_to_docs
from spacy.util import ENV_VARS, get_minor_version, load_config, load_model_from_config
from .util import make_tempdir
def test_cli_find_threshold(capsys):

    def make_examples(nlp: Language) -> List[Example]:
        docs: List[Example] = []
        for t in [('I am angry and confused in the Bank of America.', {'cats': {'ANGRY': 1.0, 'CONFUSED': 1.0, 'HAPPY': 0.0}, 'spans': {'sc': [(31, 46, 'ORG')]}}), ('I am confused but happy in New York.', {'cats': {'ANGRY': 0.0, 'CONFUSED': 1.0, 'HAPPY': 1.0}, 'spans': {'sc': [(27, 35, 'GPE')]}})]:
            doc = nlp.make_doc(t[0])
            docs.append(Example.from_dict(doc, t[1]))
        return docs

    def init_nlp(components: Tuple[Tuple[str, Dict[str, Any]], ...]=()) -> Tuple[Language, List[Example]]:
        new_nlp = English()
        new_nlp.add_pipe(factory_name='textcat_multilabel', name='tc_multi', config={'threshold': 0.9})
        for cfn, comp_config in components:
            new_nlp.add_pipe(cfn, config=comp_config)
        new_examples = make_examples(new_nlp)
        new_nlp.initialize(get_examples=lambda: new_examples)
        for i in range(5):
            new_nlp.update(new_examples)
        return (new_nlp, new_examples)
    with make_tempdir() as docs_dir:
        nlp, examples = init_nlp()
        DocBin(docs=[example.reference for example in examples]).to_disk(docs_dir / 'docs.spacy')
        with make_tempdir() as nlp_dir:
            nlp.to_disk(nlp_dir)
            best_threshold, best_score, res = find_threshold(model=nlp_dir, data_path=docs_dir / 'docs.spacy', pipe_name='tc_multi', threshold_key='threshold', scores_key='cats_macro_f', silent=True)
            assert best_score == max(res.values())
            assert res[1.0] == 0.0
        nlp, _ = init_nlp((('spancat', {}),))
        with make_tempdir() as nlp_dir:
            nlp.to_disk(nlp_dir)
            best_threshold, best_score, res = find_threshold(model=nlp_dir, data_path=docs_dir / 'docs.spacy', pipe_name='spancat', threshold_key='threshold', scores_key='spans_sc_f', silent=True)
            assert best_score == max(res.values())
            assert res[1.0] == 0.0
        nlp, _ = init_nlp((('textcat_multilabel', {}),))
        with make_tempdir() as nlp_dir:
            nlp.to_disk(nlp_dir)
            assert find_threshold(model=nlp_dir, data_path=docs_dir / 'docs.spacy', pipe_name='tc_multi', threshold_key='threshold', scores_key='cats_macro_f', silent=True)
        nlp, _ = init_nlp()
        with make_tempdir() as nlp_dir:
            nlp.to_disk(nlp_dir)
            with pytest.raises(AttributeError):
                find_threshold(model=nlp_dir, data_path=docs_dir / 'docs.spacy', pipe_name='_', threshold_key='threshold', scores_key='cats_macro_f', silent=True)