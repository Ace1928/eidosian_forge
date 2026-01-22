import sys
import os
import re
import warnings
import types
import unicodedata
def set_name_id_map(self, node, id, msgnode=None, explicit=None):
    """
        `self.nameids` maps names to IDs, while `self.nametypes` maps names to
        booleans representing hyperlink type (True==explicit,
        False==implicit).  This method updates the mappings.

        The following state transition table shows how `self.nameids` ("ids")
        and `self.nametypes` ("types") change with new input (a call to this
        method), and what actions are performed ("implicit"-type system
        messages are INFO/1, and "explicit"-type system messages are ERROR/3):

        ====  =====  ========  ========  =======  ====  =====  =====
         Old State    Input          Action        New State   Notes
        -----------  --------  -----------------  -----------  -----
        ids   types  new type  sys.msg.  dupname  ids   types
        ====  =====  ========  ========  =======  ====  =====  =====
        -     -      explicit  -         -        new   True
        -     -      implicit  -         -        new   False
        None  False  explicit  -         -        new   True
        old   False  explicit  implicit  old      new   True
        None  True   explicit  explicit  new      None  True
        old   True   explicit  explicit  new,old  None  True   [#]_
        None  False  implicit  implicit  new      None  False
        old   False  implicit  implicit  new,old  None  False
        None  True   implicit  implicit  new      None  True
        old   True   implicit  implicit  new      old   True
        ====  =====  ========  ========  =======  ====  =====  =====

        .. [#] Do not clear the name-to-id map or invalidate the old target if
           both old and new targets are external and refer to identical URIs.
           The new target is invalidated regardless.
        """
    for name in node['names']:
        if name in self.nameids:
            self.set_duplicate_name_id(node, id, name, msgnode, explicit)
        else:
            self.nameids[name] = id
            self.nametypes[name] = explicit