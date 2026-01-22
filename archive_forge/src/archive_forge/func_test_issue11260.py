import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.issue('11260')
def test_issue11260():
    annots = {'words': ['I', 'like', 'New', 'York', '.'], 'spans': {'cities': [(7, 15, 'LOC', '')], 'people': [(0, 1, 'PERSON', '')]}}
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    example = Example.from_dict(predicted, annots)
    assert len(example.reference.spans['cities']) == 1
    assert len(example.reference.spans['people']) == 1
    output_dict = example.to_dict()
    assert 'spans' in output_dict['doc_annotation']
    assert output_dict['doc_annotation']['spans']['cities'] == annots['spans']['cities']
    assert output_dict['doc_annotation']['spans']['people'] == annots['spans']['people']
    output_example = Example.from_dict(predicted, output_dict)
    assert len(output_example.reference.spans['cities']) == len(example.reference.spans['cities'])
    assert len(output_example.reference.spans['people']) == len(example.reference.spans['people'])
    for span in example.reference.spans['cities']:
        assert span.label_ == 'LOC'
        assert span.text == 'New York'
        assert span.start_char == 7
    for span in example.reference.spans['people']:
        assert span.label_ == 'PERSON'
        assert span.text == 'I'
        assert span.start_char == 0