from nltk.internals import Counter
from nltk.sem.logic import APP, LogicParser

        :param other: ``BindingDict`` The dict with which to combine self
        :return: ``BindingDict`` A new dict containing all the elements of both parameters
        :raise VariableBindingException: If the parameter dictionaries are not consistent with each other
        