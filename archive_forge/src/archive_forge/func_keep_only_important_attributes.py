from nltk.corpus import wordnet
from nltk.wsd import lesk
import spacy
def keep_only_important_attributes(parsed_sentences):
    results = []
    for parsed_sentence in parsed_sentences:
        attributes = []
        for word in parsed_sentence:
            attributes.append({'text': word.text, 'lemma': word.lemma_, 'pos': word.pos_})
        results.append(attributes)
    return results