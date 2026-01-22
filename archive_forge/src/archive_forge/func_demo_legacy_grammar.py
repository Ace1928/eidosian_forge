import codecs
from nltk.sem import evaluate
def demo_legacy_grammar():
    """
    Check that interpret_sents() is compatible with legacy grammars that use
    a lowercase 'sem' feature.

    Define 'test.fcfg' to be the following

    """
    from nltk.grammar import FeatureGrammar
    g = FeatureGrammar.fromstring("\n    % start S\n    S[sem=<hello>] -> 'hello'\n    ")
    print('Reading grammar: %s' % g)
    print('*' * 20)
    for reading in interpret_sents(['hello'], g, semkey='sem'):
        syn, sem = reading[0]
        print()
        print('output: ', sem)