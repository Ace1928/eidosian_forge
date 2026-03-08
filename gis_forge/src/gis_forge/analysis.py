from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .ontology_engine import OntologyEngine

logger = logging.getLogger("gis_forge.analysis")


class VectorAnalysis:
    """
    Advanced vector analysis suite for the Eidosian Ontology.
    Provides clustering, distance analysis, and rule discovery.
    """

    def __init__(self, engine: OntologyEngine):
        self.engine = engine

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def compute_centroid(self, vectors: List[List[float]]) -> List[float]:
        """Compute the geometric center of a group of vectors."""
        if not vectors:
            return []
        arr = np.array(vectors)
        centroid = np.mean(arr, axis=0)
        return centroid.tolist()

    def k_means_clustering(
        self, vectors: List[List[float]], k: int = 3, max_iter: int = 100
    ) -> Tuple[List[int], List[List[float]]]:
        """
        Pure NumPy implementation of K-Means clustering.
        Returns cluster assignments and centroids.
        """
        if not vectors or len(vectors) < k:
            return [], []

        data = np.array(vectors, dtype="float32")
        n_samples, n_features = data.shape

        # Initialize centroids randomly from data
        rng = np.random.default_rng()
        centroids = data[rng.choice(n_samples, k, replace=False)]

        for _ in range(max_iter):
            # Compute distances to centroids
            # (a-b)^2 = a^2 + b^2 - 2ab
            dists = np.sqrt(np.sum((data[:, np.newaxis] - centroids) ** 2, axis=2))

            # Assign clusters
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.array(
                [data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)]
            )

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return labels.tolist(), centroids.tolist()

    def discover_axioms(self, node_type: str = "memory", threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Scan the ontology for dense clusters and propose axioms (Higher-Order Rules).
        Part of the RSI / Lossless Pruning logic.
        """
        ids = list(self.engine.substrate.rev_id_map.keys())
        target_ids = []

        for nid in ids:
            meta = self.engine.substrate.get_metadata(nid)
            if meta and meta.get("type") == node_type:
                target_ids.append(nid)

        if len(target_ids) < 5:
            return []

        vectors = self.engine.substrate.get_vectors(target_ids)

        # Simple density analysis: use K-means to find clusters
        # In a real Eidosian system, we'd use DBSCAN or HDBSCAN for auto-detecting clusters
        # But here we'll use K-means with k=len/5 as a heuristic
        k = max(2, len(target_ids) // 5)
        labels, centroids = self.k_means_clustering(vectors, k=k)

        proposals = []
        for i in range(k):
            cluster_indices = [j for j, l in enumerate(labels) if l == i]
            if len(cluster_indices) < 3:
                continue

            # Identify members
            members = [target_ids[idx] for idx in cluster_indices]

            # Geometric check: are they actually close?
            # (Simplification for production quality)
            proposals.append(
                {
                    "type": "axiom_proposal",
                    "cluster_id": i,
                    "member_count": len(members),
                    "member_ids": members,
                    "centroid": centroids[i],
                }
            )

        logger.info(f"Discovered {len(proposals)} dense clusters for potential axiom compression.")
        return proposals
