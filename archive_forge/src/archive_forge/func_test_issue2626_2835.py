import re
import numpy
import pytest
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import (
from spacy.vocab import Vocab
@pytest.mark.parametrize('text', ['ABLEItemColumn IAcceptance Limits of ErrorIn-Service Limits of ErrorColumn IIColumn IIIColumn IVColumn VComputed VolumeUnder Registration of\xa0VolumeOver Registration of\xa0VolumeUnder Registration of\xa0VolumeOver Registration of\xa0VolumeCubic FeetCubic FeetCubic FeetCubic FeetCubic Feet1Up to 10.0100.0050.0100.005220.0200.0100.0200.010350.0360.0180.0360.0184100.0500.0250.0500.0255Over 100.5% of computed volume0.25% of computed volume0.5% of computed volume0.25% of computed volume TABLE ItemColumn IAcceptance Limits of ErrorIn-Service Limits of ErrorColumn IIColumn IIIColumn IVColumn VComputed VolumeUnder Registration of\xa0VolumeOver Registration of\xa0VolumeUnder Registration of\xa0VolumeOver Registration of\xa0VolumeCubic FeetCubic FeetCubic FeetCubic FeetCubic Feet1Up to 10.0100.0050.0100.005220.0200.0100.0200.010350.0360.0180.0360.0184100.0500.0250.0500.0255Over 100.5% of computed volume0.25% of computed volume0.5% of computed volume0.25% of computed volume ItemColumn IAcceptance Limits of ErrorIn-Service Limits of ErrorColumn IIColumn IIIColumn IVColumn VComputed VolumeUnder Registration of\xa0VolumeOver Registration of\xa0VolumeUnder Registration of\xa0VolumeOver Registration of\xa0VolumeCubic FeetCubic FeetCubic FeetCubic FeetCubic Feet1Up to 10.0100.0050.0100.005220.0200.0100.0200.010350.0360.0180.0360.0184100.0500.0250.0500.0255Over 100.5% of computed volume0.25% of computed volume0.5% of computed volume0.25% of computed volume', 'oow.jspsearch.eventoracleopenworldsearch.technologyoraclesolarissearch.technologystoragesearch.technologylinuxsearch.technologyserverssearch.technologyvirtualizationsearch.technologyengineeredsystemspcodewwmkmppscem:'])
@pytest.mark.issue(2626)
def test_issue2626_2835(en_tokenizer, text):
    """Check that sentence doesn't cause an infinite loop in the tokenizer."""
    doc = en_tokenizer(text)
    assert doc