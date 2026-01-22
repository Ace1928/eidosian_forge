import os, sys, datetime, re
from rdflib import Graph
from ..utils import create_file_name
from . import VocabCachingInfo
import pickle
def offline_cache_generation(args, verify=True):
    """Generate a cache for the vocabulary in args.

    @param args:   array of vocabulary URIs.
    @param verify: whether the SSL certificate needs to be verified.
    """

    class LocalOption:

        def __init__(self):
            self.vocab_cache_report = True

        def pr(self, wae, txt, warning_type, context):
            print('====')
            if warning_type != None:
                print(warning_type)
            print(wae + ': ' + txt)
            if context != None:
                print(context)
            print('====')

        def add_warning(self, txt, warning_type=None, context=None):
            """Add a warning to the processor graph.
            @param txt: the warning text. 
            @keyword warning_type: Warning Class
            @type warning_type: URIRef
            @keyword context: possible context to be added to the processor graph
            @type context: URIRef or String
            """
            self.pr('Warning', txt, warning_type, context)

        def add_info(self, txt, info_type=None, context=None):
            """Add an informational comment to the processor graph.
            @param txt: the information text. 
            @keyword info_type: Info Class
            @type info_type: URIRef
            @keyword context: possible context to be added to the processor graph
            @type context: URIRef or String
            """
            self.pr('Info', txt, info_type, context)

        def add_error(self, txt, err_type=None, context=None):
            """Add an error  to the processor graph.
            @param txt: the information text. 
            @keyword err_type: Error Class
            @type err_type: URIRef
            @keyword context: possible context to be added to the processor graph
            @type context: URIRef or String
            """
            self.pr('Error', txt, err_type, context)
    for uri in args:
        print('>>>>> Writing Cache <<<<<')
        _writ = CachedVocab(uri, options=LocalOption(), verify=verify)
        print('>>>>> Reading Cache <<<<<')
        rd = CachedVocab(uri, options=LocalOption(), verify=verify)
        print('URI: ' + uri)
        print('default vocab: ' + rd.vocabulary)
        print('terms: %s prefixes: %s' % (rd.terms, rd.ns))