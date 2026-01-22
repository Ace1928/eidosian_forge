"""
Species registry and migration logic.

The species registry holds the canonical definitions of each species in
the simulation. Each entry includes high energy (HE) properties that
govern formation and stability as well as low energy (LE) properties
used by chemistry. Species definitions are loaded from and stored to
disk to allow cross‑run reuse and accumulation. The registry also
supports migrations to add new LE properties without changing the
species IDs (which are derived solely from HE properties).
"""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional


@dataclass
class Species:
    id: str
    he_props: Dict[str, float]
    le_props: Dict[str, float]
    schema_version: int = 1
    provenance: Dict[str, object] = field(default_factory=dict)
    stability_stats: Dict[str, object] = field(default_factory=dict)


class SpeciesRegistry:
    """Persistent store for species definitions.

    Only high‑energy properties contribute to the species ID. Low energy
    properties are added via migrations and do not affect the key.
    """

    def __init__(self, path: str, he_prop_defs: List[str], le_prop_defs: List[str]):
        self.path = path
        self.he_prop_defs = he_prop_defs
        self.le_prop_defs = le_prop_defs
        self.species: Dict[str, Species] = {}
        self.schema_version: int = 1
        # load existing registry if exists
        if os.path.exists(path):
            self._load()
        else:
            # create directory if needed
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._save()

    def _load(self) -> None:
        with open(self.path, 'r') as f:
            data = json.load(f)
        self.schema_version = data.get("schema_version", 1)
        for sid, entry in data.get("species", {}).items():
            self.species[sid] = Species(
                id=sid,
                he_props=entry["he_props"],
                le_props=entry.get("le_props", {}),
                schema_version=entry.get("schema_version", 1),
                provenance=entry.get("provenance", {}),
                stability_stats=entry.get("stability_stats", {}),
            )

    def _save(self) -> None:
        data = {
            "schema_version": self.schema_version,
            "species": {
                sid: {
                    "he_props": s.he_props,
                    "le_props": s.le_props,
                    "schema_version": s.schema_version,
                    "provenance": s.provenance,
                    "stability_stats": s.stability_stats,
                }
                for sid, s in self.species.items()
            },
        }
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def quantise_props(self, he_props: Dict[str, float], bins: int = 256) -> Dict[str, int]:
        """Quantise continuous HE property values to integer bins.

        The property definitions must be known in ``he_prop_defs``. Values
        outside [0,1] are clamped before quantisation. If a property is
        missing, zero is used.
        """
        q = {}
        for name in self.he_prop_defs:
            val = he_props.get(name, 0.0)
            val = max(0.0, min(val, 1.0))  # clamp to 0..1
            q[name] = int(round(val * (bins - 1)))
        return q

    def dequantise_props(self, q: Dict[str, int], bins: int = 256) -> Dict[str, float]:
        """Inverse of quantisation: map integers back to floats in [0,1]."""
        return {k: v / (bins - 1) for k, v in q.items()}

    def _hash_props(self, q: Dict[str, int]) -> str:
        """Compute stable hash of quantised HE properties for ID."""
        # convert to tuple sorted by property name to ensure determinism
        items = tuple((k, q[k]) for k in sorted(q.keys()))
        m = hashlib.blake2s(repr(items).encode('utf-8'))
        # produce 12 char base32-like id
        return m.hexdigest()[:12]

    def get_or_create_species(self, he_props: Dict[str, float], provenance: Optional[Dict[str, object]] = None) -> Species:
        """Return existing species or create a new entry.

        Uses quantisation and hashing of HE properties to derive a unique
        identifier. If the species already exists the existing entry is
        returned. Otherwise a new Species is created with empty LE props
        initialised to default values.
        """
        q = self.quantise_props(he_props)
        sid = self._hash_props(q)
        if sid in self.species:
            return self.species[sid]
        # create new species
        deq = self.dequantise_props(q)
        # fill missing HE props with zeros (ensuring dictionary has all keys)
        full_he = {name: he_props.get(name, 0.0) for name in self.he_prop_defs}
        # initial LE props are zeros
        le = {name: 0.0 for name in self.le_prop_defs}
        s = Species(id=sid, he_props=full_he, le_props=le)
        if provenance:
            s.provenance = provenance
        self.species[sid] = s
        return s

    def get_or_create_species_quantised(
        self,
        q: Dict[str, int],
        he_props: Dict[str, float],
        provenance: Optional[Dict[str, object]] = None,
    ) -> Species:
        """Return existing species or create a new entry from quantised props."""
        sid = self._hash_props(q)
        if sid in self.species:
            return self.species[sid]
        full_he = {name: he_props.get(name, 0.0) for name in self.he_prop_defs}
        le = {name: 0.0 for name in self.le_prop_defs}
        s = Species(id=sid, he_props=full_he, le_props=le)
        if provenance:
            s.provenance = provenance
        self.species[sid] = s
        return s

    def save(self) -> None:
        self._save()

    def mark_stable(self, species_id: str, stability_data: Dict[str, object]) -> None:
        if species_id not in self.species:
            return
        self.species[species_id].stability_stats = stability_data
        # Optionally persist periodically; for now this is deferred to caller

    def migrate_le_properties(self, new_props: Dict[str, Callable[[Dict[str, float], str], float]], new_version: int) -> None:
        """Add new LE properties by computing them from HE props.

        ``new_props`` maps property names to functions that accept
        (he_props, species_id) and return a float. The registry schema
        version is incremented to ``new_version``. Existing species have
        their LE props updated accordingly. IDs remain unchanged.
        """
        for s in self.species.values():
            for name, func in new_props.items():
                s.le_props[name] = func(s.he_props, s.id)
            s.schema_version = new_version
        # update registry definitions
        for name in new_props.keys():
            if name not in self.le_prop_defs:
                self.le_prop_defs.append(name)
        self.schema_version = new_version
        self._save()
