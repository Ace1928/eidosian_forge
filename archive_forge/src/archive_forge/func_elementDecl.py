import os
import warnings
from collections import Counter
from xml.parsers import expat
from io import BytesIO
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from urllib.request import urlopen
from urllib.parse import urlparse
from Bio import StreamModeError
def elementDecl(self, name, model):
    """Call a call-back function for each element declaration in a DTD.

        This is used for each element declaration in a DTD like::

            <!ELEMENT       name          (...)>

        The purpose of this function is to determine whether this element
        should be regarded as a string, integer, list, dictionary, structure,
        or error.
        """
    if name.upper() == 'ERROR':
        self.errors.add(name)
        return
    if name == 'Item' and model == (expat.model.XML_CTYPE_MIXED, expat.model.XML_CQUANT_REP, None, ((expat.model.XML_CTYPE_NAME, expat.model.XML_CQUANT_NONE, 'Item', ()),)):
        self.items.add(name)
        return
    while model[0] in (expat.model.XML_CTYPE_SEQ, expat.model.XML_CTYPE_CHOICE) and model[1] in (expat.model.XML_CQUANT_NONE, expat.model.XML_CQUANT_OPT) and (len(model[3]) == 1):
        model = model[3][0]
    if model[0] in (expat.model.XML_CTYPE_MIXED, expat.model.XML_CTYPE_EMPTY):
        if model[1] == expat.model.XML_CQUANT_REP:
            children = model[3]
            allowed_tags = frozenset((child[2] for child in children))
        else:
            allowed_tags = frozenset()
        self.strings[name] = allowed_tags
        return
    if model == (expat.model.XML_CTYPE_ANY, expat.model.XML_CQUANT_NONE, None, ()):
        allowed_tags = None
        repeated_tags = None
        args = (allowed_tags, repeated_tags)
        self.constructors[name] = (DictionaryElement, args)
        return
    if model[0] in (expat.model.XML_CTYPE_CHOICE, expat.model.XML_CTYPE_SEQ) and model[1] in (expat.model.XML_CQUANT_PLUS, expat.model.XML_CQUANT_REP):
        children = model[3]
        allowed_tags = frozenset((child[2] for child in children))
        if model[0] == expat.model.XML_CTYPE_SEQ:
            if len(children) > 1:
                assert model[1] == expat.model.XML_CQUANT_PLUS
                first_child = children[0]
                assert first_child[1] == expat.model.XML_CQUANT_NONE
                first_tag = first_child[2]
                args = (allowed_tags, first_tag)
                self.constructors[name] = (OrderedListElement, args)
                return
            assert len(children) == 1
        self.constructors[name] = (ListElement, (allowed_tags,))
        return
    single = []
    multiple = []
    errors = []

    def count(model):
        quantifier, key, children = model[1:]
        if key is None:
            if quantifier in (expat.model.XML_CQUANT_PLUS, expat.model.XML_CQUANT_REP):
                for child in children:
                    multiple.append(child[2])
            else:
                for child in children:
                    count(child)
        elif key.upper() == 'ERROR':
            errors.append(key)
        elif quantifier in (expat.model.XML_CQUANT_NONE, expat.model.XML_CQUANT_OPT):
            single.append(key)
        elif quantifier in (expat.model.XML_CQUANT_PLUS, expat.model.XML_CQUANT_REP):
            multiple.append(key)
    count(model)
    if len(single) == 0 and len(multiple) == 1:
        allowed_tags = frozenset(multiple + errors)
        self.constructors[name] = (ListElement, (allowed_tags,))
    else:
        allowed_tags = frozenset(single + multiple + errors)
        repeated_tags = frozenset(multiple)
        args = (allowed_tags, repeated_tags)
        self.constructors[name] = (DictionaryElement, args)