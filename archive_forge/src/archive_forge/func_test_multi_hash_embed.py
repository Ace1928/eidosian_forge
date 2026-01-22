from typing import List
import numpy
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from thinc.api import (
from spacy.lang.en import English
from spacy.lang.en.examples import sentences as EN_SENTENCES
from spacy.ml.extract_spans import _get_span_indices, extract_spans
from spacy.ml.models import (
from spacy.ml.staticvectors import StaticVectors
from spacy.util import registry
def test_multi_hash_embed():
    embed = MultiHashEmbed(width=32, rows=[500, 500, 500], attrs=['NORM', 'PREFIX', 'SHAPE'], include_static_vectors=False)
    hash_embeds = [node for node in embed.walk() if node.name == 'hashembed']
    assert len(hash_embeds) == 3
    assert list(sorted((he.attrs['column'] for he in hash_embeds))) == [0, 1, 2]
    assert len(set((he.attrs['seed'] for he in hash_embeds))) == 3
    assert [he.get_dim('nV') for he in hash_embeds] == [500, 500, 500]
    embed = MultiHashEmbed(width=32, rows=[1000, 50, 250], attrs=['NORM', 'PREFIX', 'SHAPE'], include_static_vectors=False)
    hash_embeds = [node for node in embed.walk() if node.name == 'hashembed']
    assert [he.get_dim('nV') for he in hash_embeds] == [1000, 50, 250]