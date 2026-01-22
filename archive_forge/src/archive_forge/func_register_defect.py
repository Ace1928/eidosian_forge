import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
def register_defect(self, obj, defect):
    """Record 'defect' on 'obj'.

        Called by handle_defect if raise_on_defect is False.  This method is
        part of the Policy API so that Policy subclasses can implement custom
        defect handling.  The default implementation calls the append method of
        the defects attribute of obj.  The objects used by the email package by
        default that get passed to this method will always have a defects
        attribute with an append method.

        """
    obj.defects.append(defect)