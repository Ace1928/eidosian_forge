import io
import math
import os
import typing
import weakref
def set_page_labels(doc, labels):
    """Add / replace page label definitions in PDF document.

    Args:
        doc: PDF document (resp. 'self').
        labels: list of label dictionaries like:
        {'startpage': int, 'prefix': str, 'style': str, 'firstpagenum': int},
        as returned by get_page_labels().
    """

    def create_label_str(label):
        """Convert Python label dict to correspnding PDF rule string.

        Args:
            label: (dict) build rule for the label.
        Returns:
            PDF label rule string wrapped in "<<", ">>".
        """
        s = '%i<<' % label['startpage']
        if label.get('prefix', '') != '':
            s += '/P(%s)' % label['prefix']
        if label.get('style', '') != '':
            s += '/S/%s' % label['style']
        if label.get('firstpagenum', 1) > 1:
            s += '/St %i' % label['firstpagenum']
        s += '>>'
        return s

    def create_nums(labels):
        """Return concatenated string of all labels rules.

        Args:
            labels: (list) dictionaries as created by function 'rule_dict'.
        Returns:
            PDF compatible string for page label definitions, ready to be
            enclosed in PDF array 'Nums[...]'.
        """
        labels.sort(key=lambda x: x['startpage'])
        s = ''.join([create_label_str(label) for label in labels])
        return s
    doc._set_page_labels(create_nums(labels))