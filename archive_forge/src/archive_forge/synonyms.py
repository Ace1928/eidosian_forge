from nltk.corpus import wordnet
from nltk.wsd import lesk
import spacy


mapping_pos_spacy_pos_wordnet = {'VERB': wordnet.VERB,
                                 'NOUN': wordnet.NOUN,
                                 'ADJ': wordnet.ADJ,
                                 'ADV': wordnet.ADV}


def parse_sentences(sentences, nlp_tool):
    return [nlp_tool(sentence) for sentence in sentences]


def keep_only_important_attributes(parsed_sentences):
    results = []
    for parsed_sentence in parsed_sentences:
        attributes = []
        for word in parsed_sentence:
            attributes.append({'text':word.text, 'lemma':word.lemma_, 'pos':word.pos_})
        results.append(attributes)
    return results


def compute_synonyms(parsed_sentences, words_to_disambiguate_per_sentence, window_size=10, disambiguate=True):
    final_parsed_sentences = []
    words_before_after = window_size // 2

    for parsed_sentence, words_to_disambiguate in zip(parsed_sentences, words_to_disambiguate_per_sentence):
        words = [{'text':word['text'], 'lemma':word['lemma'], 'pos':mapping_pos_spacy_pos_wordnet.get(word['pos'], list(mapping_pos_spacy_pos_wordnet.values()))} for word in parsed_sentence]
        tokenized_sentence = []
        if disambiguate:
            tokenized_sentence = [word_attributes['text'] for word_attributes in words]

        for word_pos, word_attributes in enumerate(words):
            word_attributes['synonyms'] = {}
            if word_attributes['text'] in words_to_disambiguate or word_attributes['lemma'] in words_to_disambiguate:
                # Gather synsets
                word_attributes['synonyms'] = wordnet.synsets(word_attributes['lemma'])
                if len(word_attributes['synonyms']) == 0:
                    word_attributes['synonyms'] = None
                # Use context to disambiguate
                if disambiguate:
                    context_tokens = tokenized_sentence[max(word_pos - words_before_after, 0):word_pos] + tokenized_sentence[word_pos + 1:word_pos + 1 + words_before_after]
                    word_attributes['synonyms'] = lesk(context_tokens, word_attributes['lemma'], word_attributes['pos'], word_attributes['synonyms'])

                # Pick the first sense if several ones available
                if isinstance(word_attributes['synonyms'], list):
                    word_attributes['synonyms'] = word_attributes['synonyms'][0]

                # Remove the word itself
                if word_attributes['synonyms'] is not None:
                    word_attributes['synonyms'] = {synonym.replace('_', ' ') for synonym in word_attributes['synonyms'].lemma_names()}
                    word_attributes['synonyms'].discard(word_attributes['text'])
                    word_attributes['synonyms'].discard(word_attributes['lemma'])
                else:
                    word_attributes['synonyms'] = {}

        final_parsed_sentences.append(words)

    return final_parsed_sentences


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    nlp.remove_pipe('ner')
    nlp.remove_pipe('parser')

    sentences = ["Hello my friend , can you please help me ?", "Where can I find the meatball can ?"]
    words_to_disambiguate_per_sentence = [{'friend', 'can', 'help'}, {'can', 'find', 'meatball', 'can'}]
    parsed_sentences = parse_sentences(sentences, nlp)
    parsed_sentences = keep_only_important_attributes(parsed_sentences)
    parsed_sentences = compute_synonyms(parsed_sentences, words_to_disambiguate_per_sentence, window_size=10, disambiguate=False)

    for parsed_sentence in parsed_sentences:
        print(' '.join([word['text'] for word in parsed_sentence]))
        for word in parsed_sentence:
            print(' '.join(word['synonyms']))
        print()
    print()
    exit()

    words = []
    for parsed_sentence in parsed_sentences:
        words.append(lesk([word['text'] for word in parsed_sentence], 'can'))
    a = 2

    lemma = 'can'
    pos = wordnet.NOUN
    synonyms = wordnet.synsets(lemma, pos)
    b = synonyms[0].hypernyms()
    word = synonyms[0]
    synonyms = synonyms[1:]
    for synonym in synonyms:
        print('{} ({}) - {}: {} {}'.format(word.name(), word.pos, synonym.name(), word.shortest_path_distance(synonym), word.path_similarity(synonym, simulate_root=False)))

    #lowest_common_hypernyms
