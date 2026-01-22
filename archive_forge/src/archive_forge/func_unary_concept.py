import os
import re
import shelve
import sys
import nltk.data
def unary_concept(label, subj, records):
    """
    Make a unary concept out of the primary key in a record.

    A record is a list of entities in some relation, such as
    ``['france', 'paris']``, where ``'france'`` is acting as the primary
    key.

    :param label: the preferred label for the concept
    :type label: string
    :param subj: position in the record of the subject of the predicate
    :type subj: int
    :param records: a list of records
    :type records: list of lists
    :return: ``Concept`` of arity 1
    :rtype: Concept
    """
    c = Concept(label, arity=1, extension=set())
    for record in records:
        c.augment(record[subj])
    return c