"""
Handles building and updating the knowledge graph from database sources.

Encapsulates logic for fetching data, adding nodes/edges, ensuring sample data,
and verifying database integrity related to graph construction. Adheres to
Eidosian principles of modularity, precision, and structural integrity.

Architecture:
    ┌──────────────────┐      ┌──────────────────┐
    │  GraphManager    │◄────►│   GraphBuilder   │
    │ (Orchestrator)   │      │ (Data Fetching & │
    └────────┬─────────┘      │  Graph Assembly) │
             │                └──────────────────┘
             ▼
    ┌──────────────────┐
    │   DBManager      │
    │ (Database Conn)  │
    └──────────────────┘
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, cast

from eidosian_core import eidosian

# Import necessary components (adjust paths as needed)
from word_forge.exceptions import GraphDataError, GraphError
from word_forge.graph.graph_config import (
    RelationshipDimension,
    RelationshipTuple,
    RelType,
    WordId,
    WordTuple,
)
from word_forge.relationships import RelationshipProperties

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager


@dataclass(frozen=True)
class GraphUpdateMetrics:
    """Lightweight snapshot of the most recent graph update."""

    new_nodes: int = 0
    new_edges: int = 0
    processed_words: int = 0
    max_last_refreshed: float = 0.0
    full_rebuild: bool = False


class GraphBuilder:
    """
    Manages graph construction and updates from the database.

    Responsible for fetching word and relationship data, constructing the
    NetworkX graph structure, ensuring the presence of sample data if needed,
    and verifying the underlying database schema. Delegates layout computations
    to the GraphLayout module via the GraphManager.

    Attributes:
        manager: Reference to the main GraphManager for state access.
        logger: Logger instance for this module.
    """

    def __init__(self, manager: GraphManager) -> None:
        """
        Initialize the GraphBuilder with a reference to the GraphManager.

        Args:
            manager: The orchestrating GraphManager instance.
        """
        self.manager: GraphManager = manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Use config from manager for consistency
        self._config = self.manager.config
        self._metadata_key = "graph_last_refresh"
        self._last_update_metrics: GraphUpdateMetrics = GraphUpdateMetrics()
        self._last_refresh_watermark: float = self._load_last_refresh_watermark()

    @property
    def last_update_metrics(self) -> GraphUpdateMetrics:
        """Return metrics captured during the most recent update cycle."""

        return self._last_update_metrics

    def _set_last_update_metrics(
        self,
        *,
        new_nodes: int,
        new_edges: int,
        processed_words: int,
        max_last_refreshed: float,
        full_rebuild: bool,
    ) -> None:
        self._last_update_metrics = GraphUpdateMetrics(
            new_nodes=new_nodes,
            new_edges=new_edges,
            processed_words=processed_words,
            max_last_refreshed=max_last_refreshed,
            full_rebuild=full_rebuild,
        )

    def _ensure_metadata_table(self, conn: sqlite3.Connection) -> None:
        """Ensure the metadata table needed for graph watermarks exists."""

        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """)

    def _load_last_refresh_watermark(self) -> float:
        """Load the persisted watermark indicating the last processed timestamp."""

        try:
            with self.manager._db_connection() as conn:
                self._ensure_metadata_table(conn)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value FROM graph_metadata WHERE key = ?",
                    (self._metadata_key,),
                )
                row = cursor.fetchone()
                if row and row[0] is not None:
                    return float(row[0])
        except (sqlite3.Error, TypeError, ValueError) as exc:
            self.logger.debug("Unable to load graph watermark from metadata table: %s", exc)
        return 0.0

    def _persist_watermark(self, value: float) -> None:
        """Persist the latest processed timestamp for incremental updates."""

        if value <= self._last_refresh_watermark:
            return

        try:
            with self.manager._db_connection() as conn:
                self._ensure_metadata_table(conn)
                conn.execute(
                    """
                    INSERT INTO graph_metadata (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value=excluded.value,
                        updated_at=excluded.updated_at
                    """,
                    (self._metadata_key, str(value), time.time()),
                )
                conn.commit()
                self._last_refresh_watermark = value
        except sqlite3.Error as exc:
            self.logger.warning(
                "Failed to persist graph watermark: %s",
                exc,
                exc_info=self.logger.isEnabledFor(logging.DEBUG),
            )

    @eidosian()
    def build_graph(self) -> None:
        """
        Construct the graph from the database, replacing any existing graph.

        Fetches all words and relationships, adds them as nodes and edges
        to the manager's graph object, builds the term-to-ID mapping,
        calculates relationship counts, and triggers a full layout computation.

        Raises:
            GraphDataError: If fetching data from the database fails.
            GraphError: For other graph construction issues.
        """
        self.logger.info("Initiating full graph build process.")
        # Clear existing graph state held by the manager
        self.manager.g.clear()
        self.manager._term_to_id.clear()
        self.manager._positions.clear()
        self.manager._relationship_counts.clear()  # Use clear() for consistency

        try:
            words, relationships, latest_refresh = self._fetch_data()
            self.logger.info(f"Fetched {len(words)} words and {len(relationships)} relationships.")
        except sqlite3.Error as db_err:
            raise GraphDataError("Database error during data fetch.", db_err) from db_err
        except Exception as e:
            raise GraphDataError(f"Unexpected error fetching data: {e}", e) from e

        # --- Node Addition ---
        total_words = len(words)
        for idx, (word_id, term) in enumerate(words, start=1):
            # Ensure term is not None or empty before adding
            if term:
                # Add node with term and ID attributes for consistency
                self.manager.g.add_node(word_id, term=term, id=word_id)
                # Store lowercase term for case-insensitive lookup
                self.manager._term_to_id[term.lower()] = word_id
            else:
                self.logger.warning(f"Skipping node with ID {word_id} due to missing term.")
            if idx % max(total_words // 10, 1) == 0:
                self.logger.info("Node build progress: %d/%d", idx, total_words)

        # --- Edge Addition ---
        total_edges = len(relationships)
        for idx, (
            word_id,
            related_term,
            rel_type,
            dimension,
            valence,
            arousal,
        ) in enumerate(relationships, start=1):
            # Validate source node exists
            if word_id not in self.manager.g.nodes():
                self.logger.debug(f"Skipping edge from non-existent node ID {word_id}.")
                continue

            # Find target node ID (case-insensitive)
            related_id = self.manager._term_to_id.get(related_term.lower())

            # Validate target node exists
            if related_id is None or related_id not in self.manager.g.nodes():
                self.logger.debug(f"Skipping edge to non-existent term '{related_term}'.")
                continue

            # Prevent self-loops unless explicitly allowed by config (future)
            if word_id == related_id:
                self.logger.debug(f"Skipping self-loop for node ID {word_id}.")
                continue

            # Add edge with calculated properties
            self._add_relationship_edge(
                word_id,
                related_id,
                rel_type,
                dimension=dimension,
                valence=valence,
                arousal=arousal,
            )
            if idx % max(total_edges // 10, 1) == 0:
                self.logger.info("Edge build progress: %d/%d", idx, total_edges)

        self.logger.info(
            f"Graph built: {self.manager.g.number_of_nodes()} nodes, {self.manager.g.number_of_edges()} edges."
        )

        # Delegate layout computation via the manager
        if self.manager.g.number_of_nodes() > 0:
            self.logger.info("Triggering full graph layout computation.")
            self.manager.layout.compute_layout()
        else:
            self.logger.info("Graph is empty, skipping layout computation.")

        self._set_last_update_metrics(
            new_nodes=self.manager.g.number_of_nodes(),
            new_edges=self.manager.g.number_of_edges(),
            processed_words=len(words),
            max_last_refreshed=latest_refresh,
            full_rebuild=True,
        )
        self._persist_watermark(latest_refresh)

    @eidosian()
    def update_graph(self) -> int:
        """
        Incrementally update the existing graph with new data from the database.

        Fetches all data and compares against the current graph state. Adds new
        nodes and edges. Triggers an incremental layout update if changes occurred.
        If the graph is initially empty, performs a full build instead.

        Returns:
            int: The number of new nodes added during the update.

        Raises:
            GraphDataError: If fetching data from the database fails.
            GraphError: For other graph update issues.
        """
        if self.manager.g.number_of_nodes() == 0:
            self.logger.info("Graph is empty, performing initial build instead of update.")
            self.build_graph()
            return self.manager.g.number_of_nodes()

        self.logger.info("Initiating incremental graph update.")
        since = self._last_refresh_watermark if self._last_refresh_watermark > 0 else None
        if since is None:
            self.logger.debug("No persisted watermark detected; falling back to full dataset fetch.")
        try:
            all_words, all_relationships, latest_refresh = self._fetch_data(since=since)
        except sqlite3.Error as db_err:
            raise GraphDataError("Database error during data fetch for update.", db_err) from db_err
        except Exception as e:
            raise GraphDataError(f"Unexpected error fetching data for update: {e}", e) from e

        current_node_ids: Set[WordId] = set(self.manager.g.nodes())
        new_nodes_added: List[WordId] = []
        new_edges_added_count = 0

        # --- Add New Nodes ---
        for word_id, term in all_words:
            if term and word_id not in current_node_ids:
                self.manager.g.add_node(word_id, term=term, id=word_id)
                self.manager._term_to_id[term.lower()] = word_id
                new_nodes_added.append(word_id)

        new_node_count = len(new_nodes_added)

        # --- Add New Edges ---
        for (
            word_id,
            related_term,
            rel_type,
            dimension,
            valence,
            arousal,
        ) in all_relationships:
            related_id = self.manager._term_to_id.get(related_term.lower())
            # Ensure both nodes exist in the potentially updated graph
            if word_id in self.manager.g and related_id is not None and related_id in self.manager.g:
                # Check if the edge (or its reverse for undirected) already exists
                if not self.manager.g.has_edge(word_id, related_id):
                    # Prevent self-loops
                    if word_id == related_id:
                        continue
                    self._add_relationship_edge(
                        word_id,
                        related_id,
                        rel_type,
                        dimension=dimension,
                        valence=valence,
                        arousal=arousal,
                    )
                    new_edges_added_count += 1

        # --- Post-Update Actions ---
        if new_node_count > 0 or new_edges_added_count > 0:
            self.logger.info(f"Graph updated: Added {new_node_count} nodes and {new_edges_added_count} edges.")
            # Delegate incremental layout update only if nodes were added
            if new_node_count > 0:
                self.logger.info("Triggering incremental layout update.")
                self.manager.layout.update_layout_incrementally(new_nodes_added)
            else:
                self.logger.info("Only edges added, layout update may not be necessary depending on algorithm.")
                # Optionally trigger full re-layout if edge changes significantly impact structure
                # self.manager.layout.compute_layout()
        else:
            self.logger.info("Graph update: No new nodes or edges detected.")

        self._set_last_update_metrics(
            new_nodes=new_node_count,
            new_edges=new_edges_added_count,
            processed_words=len(all_words),
            max_last_refreshed=latest_refresh,
            full_rebuild=False,
        )
        self._persist_watermark(latest_refresh)

        return new_node_count

    def _add_relationship_edge(
        self,
        source_id: WordId,
        target_id: WordId,
        rel_type: RelType,
        *,
        dimension: Optional[RelationshipDimension] = None,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
    ) -> None:
        """
        Adds a single relationship edge to the graph with calculated attributes.

        Internal helper method to encapsulate edge creation logic, including
        determining dimension, properties, and updating counts.

        Args:
            source_id: The ID of the source node.
            target_id: The ID of the target node.
            rel_type: The type of the relationship.
        """
        # Use manager's helper to get properties and determine dimension
        # Provide a default RelationshipProperties if type is unknown
        rel_props: RelationshipProperties = self.manager._get_relationship_properties(rel_type)
        resolved_dimension = dimension or self.manager._determine_dimension(rel_type)

        # Safely get term text for title, providing defaults
        node_attrs = dict(self.manager.g.nodes(data=True))
        source_term_text = node_attrs.get(source_id, {}).get("term", f"ID:{source_id}")
        target_term_text = node_attrs.get(target_id, {}).get("term", f"ID:{target_id}")

        # Construct edge attributes
        edge_attrs = {
            "relationship": rel_type,
            "weight": rel_props.get("weight", 1.0),  # Default weight if missing
            "color": rel_props.get(
                "color", self._config.relationship_colors.get("default", "#aaaaaa")
            ),  # Default color
            "bidirectional": rel_props.get("bidirectional", False),  # Default directionality
            "dimension": resolved_dimension,
            # Ensure title generation handles potential None terms gracefully
            "title": f"{rel_type}: {source_term_text or '?'} {'↔' if rel_props.get('bidirectional', False) else '→'} {target_term_text or '?'}",
        }

        if valence is not None:
            edge_attrs["valence"] = valence
        if arousal is not None:
            edge_attrs["arousal"] = arousal

        # Add the edge to the manager's graph
        self.manager.g.add_edge(source_id, target_id, **edge_attrs)

        # Update relationship counts held by the manager
        # Use Counter's update method for clarity
        self.manager._relationship_counts.update([rel_type])

    def _fetch_data(self, since: Optional[float] = None) -> Tuple[List[WordTuple], List[RelationshipTuple], float]:
        """
        Fetch words and relationships from the database.

        Uses the manager's database connection context manager for safe access.
        When ``since`` is provided, only rows newer than the watermark are
        returned.

        Returns:
            Tuple containing a list of word tuples (id, term) and a list of
            relationship tuples (word_id, related_term, relationship_type),
            along with the highest ``last_refreshed`` timestamp observed.

        Raises:
            GraphDataError: If database tables are missing or query fails.
            sqlite3.Error: For underlying database connection or query errors.
        """
        self.logger.debug("Fetching graph data from database.")
        words: List[WordTuple] = []
        relationships: List[RelationshipTuple] = []
        latest_refresh = self._last_refresh_watermark

        try:
            # Use manager's context manager for connection safety
            with self.manager._db_connection() as conn:
                cursor = conn.cursor()

                # Verify 'words' table exists
                cursor.execute(self._config.sql_templates["check_words_table"])
                if not cursor.fetchone():
                    raise GraphDataError("Database table 'words' not found. Cannot build graph.")

                # Fetch words: Ensure id and term are not NULL
                word_query_key = "fetch_words_since" if since is not None else "fetch_all_words"
                word_query = self._config.sql_templates.get(word_query_key)
                if word_query is None:
                    raise GraphDataError(f"SQL template '{word_query_key}' is missing.")
                params: Tuple[float, ...] = (since,) if since is not None else tuple()
                cursor.execute(word_query, params)
                words_raw = cursor.fetchall()
                words = [
                    cast(WordTuple, (row["id"], row["term"]))
                    for row in words_raw
                    if row["id"] is not None and row["term"] is not None
                ]
                for row in words_raw:
                    try:
                        refreshed = float(row["last_refreshed"] or 0.0)
                        latest_refresh = max(latest_refresh, refreshed)
                    except (TypeError, ValueError):
                        continue
                self.logger.debug(f"Fetched {len(words)} valid word entries.")

                # Verify 'relationships' table exists
                cursor.execute(self._config.sql_templates["check_relationships_table"])
                if not cursor.fetchone():
                    self.logger.warning("Database table 'relationships' not found. Graph will have no edges.")
                    # Return words only if relationships table is missing
                    return words, []

                # Fetch relationships: Ensure all parts are not NULL
                rel_query_key = "fetch_relationships_since" if since is not None else "fetch_all_relationships"
                rel_query = self._config.sql_templates.get(rel_query_key)
                rel_params: Tuple[float, ...] = (since,) if rel_query_key == "fetch_relationships_since" else tuple()
                if rel_query is None:
                    rel_query = self._config.sql_templates["fetch_all_relationships"]
                    rel_params = tuple()
                cursor.execute(rel_query, rel_params)
                relationships_raw = cursor.fetchall()
                lexical_relationships: List[RelationshipTuple] = [
                    cast(
                        RelationshipTuple,
                        (
                            row["word_id"],
                            row["related_term"],
                            row["relationship_type"],
                            "lexical",
                            None,
                            None,
                        ),
                    )
                    for row in relationships_raw
                    if row["word_id"] is not None
                    and row["related_term"] is not None
                    and row["relationship_type"] is not None
                ]
                relationships.extend(lexical_relationships)

                # Fetch emotional relationships when the table exists
                emotional_count = 0
                try:
                    emotional_query_key = (
                        "get_emotional_relationships_since" if since is not None else "get_all_emotional_relationships"
                    )
                    emotional_query = self._config.sql_templates.get(emotional_query_key)
                    if emotional_query is None:
                        emotional_query = self._config.sql_templates["get_all_emotional_relationships"]
                        emotional_params: Tuple[float, ...] = tuple()
                    else:
                        emotional_params = (
                            (since,) if emotional_query_key == "get_emotional_relationships_since" else tuple()
                        )
                    cursor.execute(emotional_query, emotional_params)
                    emotional_rows = cursor.fetchall()
                    emotional_relationships: List[RelationshipTuple] = [
                        cast(
                            RelationshipTuple,
                            (
                                row["word_id"],
                                row["related_term"],
                                row["relationship_type"],
                                "emotional",
                                row["valence"],
                                row["arousal"],
                            ),
                        )
                        for row in emotional_rows
                        if row["word_id"] is not None
                        and row["related_term"] is not None
                        and row["relationship_type"] is not None
                    ]
                    relationships.extend(emotional_relationships)
                    emotional_count = len(emotional_relationships)
                except sqlite3.Error as emotional_err:
                    self.logger.debug("Emotional relationships unavailable: %s", emotional_err)

                self.logger.debug(
                    "Fetched %d lexical and %d emotional relationship entries.",
                    len(lexical_relationships),
                    emotional_count,
                )

                return words, relationships, latest_refresh
        except sqlite3.Error as db_err:
            # Log specific DB error and re-raise as GraphDataError
            self.logger.error(f"Database query failed during data fetch: {db_err}")
            raise GraphDataError(f"Failed to fetch graph data: {db_err}", db_err) from db_err
        except Exception as e:
            # Catch any other unexpected errors during fetch
            self.logger.error(f"Unexpected error during data fetch: {e}")
            raise GraphDataError(f"An unexpected error occurred while fetching graph data: {e}", e) from e

    @eidosian()
    def ensure_sample_data(self) -> bool:
        """
        Ensure the database contains sample data if it's currently empty.

        Checks if the 'words' table has any entries. If not, it attempts to
        insert predefined sample words and relationships from the configuration.

        Returns:
            bool: True if sample data was added, False otherwise.

        Raises:
            GraphError: If adding sample data fails due to database issues.
        """
        self.logger.debug("Checking for existing data before potentially adding samples.")
        try:
            # Check word count first as a proxy for existing data
            with self.manager._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM words")
                count_result = cursor.fetchone()
                # Ensure fetchone result is not None before accessing index
                count = count_result[0] if count_result else 0
                if count > 0:
                    self.logger.info(f"Database already contains {count} words. Skipping sample data insertion.")
                    return False
        except sqlite3.Error as e:
            # Log error but proceed to attempt sample data insertion, as table might be missing
            self.logger.warning(
                f"Could not check for existing data (table might be missing): {e}. Attempting sample data insertion."
            )
        except Exception as e:
            self.logger.error(f"Unexpected error checking for existing data: {e}", exc_info=True)
            # Decide whether to proceed or raise based on policy
            raise GraphError("Failed to verify existing data presence.", e) from e

        self.logger.info("Database appears empty or 'words' table missing. Attempting to add sample data.")
        try:
            # DBManager.create_tables handles its own connection management
            self.manager.db_manager.create_tables()  # Call without passing conn

            with self.manager._db_connection() as conn:
                cursor = conn.cursor()
                # Ensure tables exist first (idempotent operation) - Moved outside the 'with' block

                inserted_word_ids: Dict[str, WordId] = {}
                # Use sample data from config - Ensure config has sample_words
                sample_words = getattr(self._config, "sample_words", [])
                sample_relationships = self._config.sample_relationships

                # --- Insert Sample Words ---
                self.logger.debug(f"Inserting {len(sample_words)} sample words.")
                for word_data in sample_words:
                    term = word_data.get("term")
                    if not term:
                        self.logger.warning(f"Skipping sample word due to missing 'term': {word_data}")
                        continue

                    try:
                        cursor.execute(
                            self._config.sql_templates["insert_sample_word"],
                            (
                                term,
                                word_data.get("definition", ""),
                                word_data.get("part_of_speech", ""),
                            ),
                        )
                        inserted_id = cursor.lastrowid
                        # Fetch ID if lastrowid is not reliable (e.g., certain SQLite versions/configs)
                        if inserted_id is None or inserted_id == 0:  # Check for 0 as well
                            cursor.execute("SELECT id FROM words WHERE term = ?", (term,))
                            row = cursor.fetchone()
                            if row:
                                inserted_id = row[0]
                        if inserted_id is not None:
                            inserted_word_ids[term.lower()] = inserted_id
                            self.logger.debug(f"Inserted sample word '{term}' with ID {inserted_id}.")
                        else:
                            self.logger.error(f"Failed to retrieve ID for inserted sample word '{term}'.")

                    except sqlite3.IntegrityError:
                        # Word likely already exists, fetch its ID
                        self.logger.debug(  # Changed to debug as it's expected if run multiple times
                            f"Sample word '{term}' already exists. Fetching ID."
                        )
                        cursor.execute("SELECT id FROM words WHERE term = ?", (term,))
                        row = cursor.fetchone()
                        if row:
                            inserted_word_ids[term.lower()] = row[0]
                        else:
                            # This case is problematic - log error
                            self.logger.error(
                                f"Sample word '{term}' reported as existing (IntegrityError) but failed to retrieve its ID."
                            )
                    except sqlite3.Error as insert_err:
                        self.logger.error(f"Database error inserting sample word '{term}': {insert_err}")

                # --- Insert Sample Relationships ---
                self.logger.debug(f"Inserting {len(sample_relationships)} sample relationships.")
                for rel_data in sample_relationships:
                    # Ensure rel_data is a tuple/list of expected length
                    if not isinstance(rel_data, (tuple, list)) or len(rel_data) != 3:
                        self.logger.warning(f"Skipping malformed sample relationship data: {rel_data}")
                        continue

                    term1, term2, rel_type = rel_data
                    if not all([term1, term2, rel_type]):
                        self.logger.warning(f"Skipping sample relationship due to missing data: {rel_data}")
                        continue

                    id1 = inserted_word_ids.get(term1.lower())
                    # We need the ID of term1, but relate to term2 text
                    # Check if term2 exists to avoid inserting relationships to non-existent sample words
                    id2_check = inserted_word_ids.get(term2.lower())

                    if id1 is not None and id2_check is not None:  # Check both terms were successfully added/found
                        try:
                            cursor.execute(
                                self._config.sql_templates["insert_sample_relationship"],
                                (
                                    id1,
                                    term2,  # Insert using term2 text as per schema
                                    rel_type,
                                ),
                            )
                            self.logger.debug(f"Inserted sample relationship: {term1} -> {term2} ({rel_type}).")
                        except sqlite3.IntegrityError:
                            self.logger.debug(  # Changed to debug
                                f"Sample relationship {term1}-{rel_type}-{term2} already exists."
                            )
                        except sqlite3.Error as rel_err:
                            self.logger.error(
                                f"Database error inserting sample relationship {term1}-{rel_type}-{term2}: {rel_err}"
                            )
                    else:
                        missing = []
                        if id1 is None:
                            missing.append(f"'{term1}' (source)")
                        if id2_check is None:
                            missing.append(f"'{term2}' (target)")
                        self.logger.warning(
                            f"Could not insert sample relationship {term1}-{rel_type}-{term2} due to missing ID(s) for: {', '.join(missing)}"
                        )

                conn.commit()
                self.logger.info("Successfully added sample data to the database.")
                return True
        except sqlite3.Error as e:
            self.logger.error(f"Failed to add sample data due to database error: {e}", exc_info=True)
            raise GraphError(f"Failed to add sample data: {e}", e) from e
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while adding sample data: {e}",
                exc_info=True,
            )
            raise GraphError(f"An unexpected error occurred while adding sample data: {e}", e) from e

    @eidosian()
    def verify_database_tables(self) -> bool:
        """
        Verify that required database tables ('words', 'relationships') exist.

        Returns:
            bool: True if both required tables exist, False otherwise.
        """
        self.logger.debug("Verifying essential database tables.")
        try:
            with self.manager._db_connection() as conn:
                cursor = conn.cursor()
                # Check for 'words' table
                cursor.execute(self._config.sql_templates["check_words_table"])
                words_exists = cursor.fetchone() is not None
                # Check for 'relationships' table
                cursor.execute(self._config.sql_templates["check_relationships_table"])
                relationships_exists = cursor.fetchone() is not None

                if words_exists and relationships_exists:
                    self.logger.debug("Database tables 'words' and 'relationships' verified.")
                    return True
                elif words_exists:
                    self.logger.warning("Database table 'words' exists, but 'relationships' is missing.")
                    return False  # Or True depending on whether relationships are optional
                else:
                    self.logger.error("Essential database table 'words' is missing.")
                    return False
        except sqlite3.Error as e:
            self.logger.error(f"Database error during table verification: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during database table verification: {e}")
            return False
