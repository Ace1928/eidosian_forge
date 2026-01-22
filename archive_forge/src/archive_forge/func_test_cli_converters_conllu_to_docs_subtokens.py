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
def test_cli_converters_conllu_to_docs_subtokens():
    lines = ['1\tDommer\tdommer\tNOUN\t_\tDefinite=Ind|Gender=Masc|Number=Sing\t2\tappos\t_\tname=O', '2-3\tFE\t_\t_\t_\t_\t_\t_\t_\t_', '2\tFinn\tFinn\tPROPN\t_\tGender=Masc\t4\tnsubj\t_\tname=B-PER', '3\tEilertsen\tEilertsen\tX\t_\tGender=Fem|Tense=past\t2\tname\t_\tname=I-PER', '4\tavstår\tavstå\tVERB\t_\tMood=Ind|Tense=Pres|VerbForm=Fin\t0\troot\t_\tSpaceAfter=No|name=O', '5\t.\t$.\tPUNCT\t_\t_\t4\tpunct\t_\tname=O']
    input_data = '\n'.join(lines)
    converted_docs = list(conllu_to_docs(input_data, n_sents=1, merge_subtokens=True, append_morphology=True))
    assert len(converted_docs) == 1
    converted = [docs_to_json(converted_docs)]
    assert converted[0]['id'] == 0
    assert len(converted[0]['paragraphs']) == 1
    assert converted[0]['paragraphs'][0]['raw'] == 'Dommer FE avstår. '
    assert len(converted[0]['paragraphs'][0]['sentences']) == 1
    sent = converted[0]['paragraphs'][0]['sentences'][0]
    assert len(sent['tokens']) == 4
    tokens = sent['tokens']
    assert [t['orth'] for t in tokens] == ['Dommer', 'FE', 'avstår', '.']
    assert [t['tag'] for t in tokens] == ['NOUN__Definite=Ind|Gender=Masc|Number=Sing', 'PROPN_X__Gender=Fem,Masc|Tense=past', 'VERB__Mood=Ind|Tense=Pres|VerbForm=Fin', 'PUNCT']
    assert [t['pos'] for t in tokens] == ['NOUN', 'PROPN', 'VERB', 'PUNCT']
    assert [t['morph'] for t in tokens] == ['Definite=Ind|Gender=Masc|Number=Sing', 'Gender=Fem,Masc|Tense=past', 'Mood=Ind|Tense=Pres|VerbForm=Fin', '']
    assert [t['lemma'] for t in tokens] == ['dommer', 'Finn Eilertsen', 'avstå', '$.']
    assert [t['head'] for t in tokens] == [1, 1, 0, -1]
    assert [t['dep'] for t in tokens] == ['appos', 'nsubj', 'ROOT', 'punct']
    ent_offsets = [(e[0], e[1], e[2]) for e in converted[0]['paragraphs'][0]['entities']]
    biluo_tags = offsets_to_biluo_tags(converted_docs[0], ent_offsets, missing='O')
    assert biluo_tags == ['O', 'U-PER', 'O', 'O']