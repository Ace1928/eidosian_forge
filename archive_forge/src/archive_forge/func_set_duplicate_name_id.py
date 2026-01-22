import sys
import os
import re
import warnings
import types
import unicodedata
def set_duplicate_name_id(self, node, id, name, msgnode, explicit):
    old_id = self.nameids[name]
    old_explicit = self.nametypes[name]
    self.nametypes[name] = old_explicit or explicit
    if explicit:
        if old_explicit:
            level = 2
            if old_id is not None:
                old_node = self.ids[old_id]
                if 'refuri' in node:
                    refuri = node['refuri']
                    if old_node['names'] and 'refuri' in old_node and (old_node['refuri'] == refuri):
                        level = 1
                if level > 1:
                    dupname(old_node, name)
                    self.nameids[name] = None
            msg = self.reporter.system_message(level, 'Duplicate explicit target name: "%s".' % name, backrefs=[id], base_node=node)
            if msgnode != None:
                msgnode += msg
            dupname(node, name)
        else:
            self.nameids[name] = id
            if old_id is not None:
                old_node = self.ids[old_id]
                dupname(old_node, name)
    else:
        if old_id is not None and (not old_explicit):
            self.nameids[name] = None
            old_node = self.ids[old_id]
            dupname(old_node, name)
        dupname(node, name)
    if not explicit or (not old_explicit and old_id is not None):
        msg = self.reporter.info('Duplicate implicit target name: "%s".' % name, backrefs=[id], base_node=node)
        if msgnode != None:
            msgnode += msg