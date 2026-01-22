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
def test_cli_converters_conllu_to_docs():
    lines = ['1\tDommer\tdommer\tNOUN\t_\tDefinite=Ind|Gender=Masc|Number=Sing\t2\tappos\t_\tO', '2\tFinn\tFinn\tPROPN\t_\tGender=Masc\t4\tnsubj\t_\tB-PER', '3\tEilertsen\tEilertsen\tPROPN\t_\t_\t2\tname\t_\tI-PER', '4\tavstår\tavstå\tVERB\t_\tMood=Ind|Tense=Pres|VerbForm=Fin\t0\troot\t_\tO']
    input_data = '\n'.join(lines)
    converted_docs = list(conllu_to_docs(input_data, n_sents=1))
    assert len(converted_docs) == 1
    converted = [docs_to_json(converted_docs)]
    assert converted[0]['id'] == 0
    assert len(converted[0]['paragraphs']) == 1
    assert len(converted[0]['paragraphs'][0]['sentences']) == 1
    sent = converted[0]['paragraphs'][0]['sentences'][0]
    assert len(sent['tokens']) == 4
    tokens = sent['tokens']
    assert [t['orth'] for t in tokens] == ['Dommer', 'Finn', 'Eilertsen', 'avstår']
    assert [t['tag'] for t in tokens] == ['NOUN', 'PROPN', 'PROPN', 'VERB']
    assert [t['head'] for t in tokens] == [1, 2, -1, 0]
    assert [t['dep'] for t in tokens] == ['appos', 'nsubj', 'name', 'ROOT']
    ent_offsets = [(e[0], e[1], e[2]) for e in converted[0]['paragraphs'][0]['entities']]
    biluo_tags = offsets_to_biluo_tags(converted_docs[0], ent_offsets, missing='O')
    assert biluo_tags == ['O', 'B-PER', 'L-PER', 'O']