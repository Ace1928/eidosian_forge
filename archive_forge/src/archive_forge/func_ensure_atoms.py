import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
@fasteners.write_locked
def ensure_atoms(self, atoms):
    """Ensure there is an atomdetail for **each** of the given atoms.

        Returns list of atomdetail uuids for each atom processed.
        """
    atom_ids = []
    missing_ads = []
    for i, atom in enumerate(atoms):
        match = misc.match_type(atom, self._ensure_matchers)
        if not match:
            raise TypeError("Unknown atom '%s' (%s) requested to ensure" % (atom, type(atom)))
        atom_detail_cls, kind = match
        atom_name = atom.name
        if not atom_name:
            raise ValueError('%s name must be non-empty' % kind)
        try:
            atom_id = self._atom_name_to_uuid[atom_name]
        except KeyError:
            missing_ads.append((i, atom, atom_detail_cls))
            atom_ids.append(None)
        else:
            ad = self._flowdetail.find(atom_id)
            if not isinstance(ad, atom_detail_cls):
                raise exceptions.Duplicate("Atom detail '%s' already exists in flow detail '%s'" % (atom_name, self._flowdetail.name))
            else:
                atom_ids.append(ad.uuid)
                self._set_result_mapping(atom_name, atom.save_as)
    if missing_ads:
        needs_to_be_created_ads = []
        for i, atom, atom_detail_cls in missing_ads:
            ad = self._create_atom_detail(atom.name, atom_detail_cls, atom_version=misc.get_version_string(atom))
            needs_to_be_created_ads.append((i, atom, ad))
        source, clone = self._fetch_flowdetail(clone=True)
        for _i, _atom, ad in needs_to_be_created_ads:
            clone.add(ad)
        self._with_connection(self._save_flow_detail, source, clone)
        for i, atom, ad in needs_to_be_created_ads:
            atom_name = atom.name
            atom_ids[i] = ad.uuid
            self._atom_name_to_uuid[atom_name] = ad.uuid
            self._set_result_mapping(atom_name, atom.save_as)
            self._failures.setdefault(atom_name, {})
    return atom_ids