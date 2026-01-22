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
@pytest.mark.issue(4665)
def test_cli_converters_conllu_empty_heads_ner():
    """
    conllu_to_docs should not raise an exception if the HEAD column contains an
    underscore
    """
    input_data = '\n1\t[\t_\tPUNCT\t-LRB-\t_\t_\tpunct\t_\t_\n2\tThis\t_\tDET\tDT\t_\t_\tdet\t_\t_\n3\tkilling\t_\tNOUN\tNN\t_\t_\tnsubj\t_\t_\n4\tof\t_\tADP\tIN\t_\t_\tcase\t_\t_\n5\ta\t_\tDET\tDT\t_\t_\tdet\t_\t_\n6\trespected\t_\tADJ\tJJ\t_\t_\tamod\t_\t_\n7\tcleric\t_\tNOUN\tNN\t_\t_\tnmod\t_\t_\n8\twill\t_\tAUX\tMD\t_\t_\taux\t_\t_\n9\tbe\t_\tAUX\tVB\t_\t_\taux\t_\t_\n10\tcausing\t_\tVERB\tVBG\t_\t_\troot\t_\t_\n11\tus\t_\tPRON\tPRP\t_\t_\tiobj\t_\t_\n12\ttrouble\t_\tNOUN\tNN\t_\t_\tdobj\t_\t_\n13\tfor\t_\tADP\tIN\t_\t_\tcase\t_\t_\n14\tyears\t_\tNOUN\tNNS\t_\t_\tnmod\t_\t_\n15\tto\t_\tPART\tTO\t_\t_\tmark\t_\t_\n16\tcome\t_\tVERB\tVB\t_\t_\tacl\t_\t_\n17\t.\t_\tPUNCT\t.\t_\t_\tpunct\t_\t_\n18\t]\t_\tPUNCT\t-RRB-\t_\t_\tpunct\t_\t_\n'
    docs = list(conllu_to_docs(input_data))
    assert not all([t.head.i for t in docs[0]])
    assert not docs[0].has_annotation('ENT_IOB')