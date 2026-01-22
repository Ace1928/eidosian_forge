import re
from nltk.stem.api import StemmerI
def parseRules(self, rule_tuple=None):
    """Validate the set of rules used in this stemmer.

        If this function is called as an individual method, without using stem
        method, rule_tuple argument will be compiled into self.rule_dictionary.
        If this function is called within stem, self._rule_tuple will be used.

        """
    rule_tuple = rule_tuple if rule_tuple else self._rule_tuple
    valid_rule = re.compile('^[a-z]+\\*?\\d[a-z]*[>\\.]?$')
    self.rule_dictionary = {}
    for rule in rule_tuple:
        if not valid_rule.match(rule):
            raise ValueError(f'The rule {rule} is invalid')
        first_letter = rule[0:1]
        if first_letter in self.rule_dictionary:
            self.rule_dictionary[first_letter].append(rule)
        else:
            self.rule_dictionary[first_letter] = [rule]