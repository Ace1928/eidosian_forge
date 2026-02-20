from eidosian_core import eidosian

"""
Demonstration of GraphManager functionality, adhering to Eidosian principles.
Ensures clarity, precision, and comprehensive testing of graph capabilities.
"""

import logging  # Import logging
import time
import traceback
from pathlib import Path  # Import Path
from typing import Dict, List, Tuple, Union, cast

import networkx as nx

from word_forge.config import config  # Import global config
from word_forge.database.database_manager import DBManager
from word_forge.exceptions import (
    GraphAnalysisError,
    GraphError,
    GraphVisualizationError,
    NodeNotFoundError,
)
from word_forge.graph.graph_manager import GraphManager

# Define type aliases for clarity
ValenceAnalysisResult = Dict[str, Union[float, int, List[Tuple[str, float]]]]

# Configure logging for the demo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("graph_demo")


@eidosian()
def graph_demo() -> None:
    """
    Demonstrate key functionality of the GraphManager class.

    Showcases graph operations, analysis, and visualization, ensuring
    robustness and adherence to configured settings. Serves as a functional
    demonstration and integration test.

    Raises:
        GraphError: If core graph operations fail.
        NodeNotFoundError: If specific terms required for demo phases are absent.
        GraphVisualizationError: If visualization generation encounters issues.

    Example:
        ```python
        # Run the comprehensive demonstration
        graph_demo()
        ```
    """

    start_time = time.time()
    logger.info("Starting GraphManager demonstration...")

    # Define output directory relative to project structure or config
    # Using config for consistency
    output_dir = Path(config.graph.default_export_path).parent / "demo_outputs"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured demo output directory exists: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create demo output directory {output_dir}: {e}")
        return  # Cannot proceed without output directory

    # Initialize database and graph managers
    # Use in-memory DB for isolated demo run, or configured path
    # db_path = ":memory:"
    db_path = config.database.db_path
    logger.info(f"Using database: {db_path}")
    db_manager = DBManager(db_path=db_path)
    graph_manager = GraphManager(db_manager)

    try:
        # --- Setup Phase ---
        logger.info("--- SETUP PHASE ---")
        # Create database tables if they don't exist
        db_manager.create_tables()

        # Check if DB has data, add sample data if empty
        if graph_manager.ensure_sample_data():
            logger.info("Added sample data to empty database.")

        # Build the graph from database
        logger.info("Building lexical graph from database...")
        graph_manager.build_graph()

        # Display graph information
        nodes_count = graph_manager.get_node_count()
        edges_count = graph_manager.get_edge_count()
        logger.info(f"Graph built with {nodes_count} nodes and {edges_count} relationships.")

        if nodes_count == 0:
            logger.error("Graph is empty after build. Cannot proceed with demo.")
            return

        # Display detailed graph summary using the manager's method
        graph_manager.display_graph_summary()

        # --- Phase 1: Basic Relationship Analysis ---
        logger.info("--- PHASE 1: BASIC RELATIONSHIP ANALYSIS ---")

        # Get related terms example - Choose a term likely in sample data
        example_term = "algorithm"
        try:
            related_terms = graph_manager.get_related_terms(example_term)
            logger.info(f"Terms related to '{example_term}': {related_terms}")

            # Filter by relationship type
            synonyms = graph_manager.get_related_terms(example_term, rel_type="synonym")
            logger.info(f"Synonyms of '{example_term}': {synonyms}")

            # Get other relationship types if available
            for rel_type in ["antonym", "hypernym", "hyponym", "domain", "function"]:
                try:
                    terms = graph_manager.get_related_terms(example_term, rel_type=rel_type)
                    if terms:
                        logger.info(f"{rel_type.capitalize()}s of '{example_term}': {terms}")
                except Exception as e:
                    logger.debug(f"Could not get {rel_type} for '{example_term}': {e}")  # Log as debug

        except NodeNotFoundError as e:
            logger.warning(f"{e}. Attempting alternative terms.")
            # Try an alternative term from the sample data
            alternative_terms = ["data", "computer", "software", "function", "graph"]
            found_alternative = False
            for alt_term in alternative_terms:
                try:
                    related_terms = graph_manager.get_related_terms(alt_term)
                    logger.info(f"Using alternative term '{alt_term}'. Related: {related_terms}")
                    example_term = alt_term  # Update for later use
                    found_alternative = True
                    break
                except NodeNotFoundError:
                    continue
            if not found_alternative:
                logger.error("Could not find any suitable example terms in the graph. Aborting phase.")
                # Decide whether to continue demo or stop
                # return

        # --- Phase 2: Multidimensional Relationship Analysis ---
        logger.info("--- PHASE 2: MULTIDIMENSIONAL RELATIONSHIP ANALYSIS ---")

        logger.info("Analyzing multidimensional relationship patterns...")
        relationship_analysis = graph_manager.analyze_multidimensional_relationships()

        # Display dimension statistics
        logger.info("Relationship dimensions found:")
        dimensions_data = relationship_analysis.get("dimensions", {})
        if dimensions_data:
            for dimension, count in dimensions_data.items():
                logger.info(f"  - {dimension}: {count} relationships")
        else:
            logger.info("  No specific dimensions found.")

        # Display multi-dimensional nodes
        multi_dim_nodes = relationship_analysis.get("multi_dimensional_nodes", {})
        if multi_dim_nodes:
            logger.info("Sample terms involved in multiple relationship dimensions:")
            for term, data in list(multi_dim_nodes.items())[:5]:  # Show first 5
                dimensions = data.get("dimensions", [])
                logger.info(f"  - {term}: {', '.join(dimensions)}")
        else:
            logger.info("  No nodes found participating in multiple dimensions.")

        # Display most common relationship types
        most_common = relationship_analysis.get("most_common", {})
        if most_common:
            logger.info("Most common relationship types per dimension:")
            for dimension, types in most_common.items():
                if types:  # types is List[Tuple[RelType, int]]
                    logger.info(f"  - {dimension}: {types[0][0]} ({types[0][1]} occurrences)")
        else:
            logger.info("  Could not determine most common relationship types.")

        # --- Phase 3: Emotional Relationship Analysis ---
        logger.info("--- PHASE 3: EMOTIONAL RELATIONSHIP ANALYSIS ---")
        # Add sample emotional data if needed for demo robustness
        logger.info("Ensuring sample emotional data exists...")
        added_emotion = False
        if "emotional" not in dimensions_data:  # Check if emotional dimension exists
            sample_emotional_relations = [
                (
                    "joy",
                    "happiness",
                    "emotional_synonym",
                    0.9,
                    0.7,
                    0.8,
                ),  # src, tgt, type, weight, src_val, tgt_val
                ("sadness", "grief", "emotional_synonym", 0.8, -0.7, -0.8),
                ("anger", "rage", "intensifies", 0.7, -0.6, -0.9),
                ("fear", "anxiety", "related_emotion", 0.6, -0.8, -0.7),
            ]
            for (
                source,
                target,
                rel_type,
                weight,
                src_val,
                tgt_val,
            ) in sample_emotional_relations:
                try:
                    source_id = graph_manager.add_word_node(source, attributes={"valence": src_val})
                    target_id = graph_manager.add_word_node(target, attributes={"valence": tgt_val})
                    graph_manager.add_relationship(
                        source_id,
                        target_id,
                        relationship=rel_type,
                        dimension="emotional",
                        weight=weight,
                        color="#ff69b4",  # Pinkish
                    )
                    added_emotion = True
                except Exception as add_err:
                    logger.warning(f"Could not add sample emotional relation {source}-{target}: {add_err}")
            if added_emotion:
                logger.info("Added sample emotional relationships.")

        # Analyze emotional valence distribution
        logger.info("Analyzing emotional valence distribution...")
        try:
            # Explicitly cast for type safety based on defined alias
            valence_analysis: ValenceAnalysisResult = cast(
                ValenceAnalysisResult,
                graph_manager.analyze_emotional_valence_distribution(dimension="emotional"),
            )

            # Check count robustly
            count_val = valence_analysis.get("count")
            if isinstance(count_val, int) and count_val > 0:
                mean_valence = cast(float, valence_analysis.get("mean", 0.0))
                # Ensure range is a tuple/list of floats
                valence_range_raw = valence_analysis.get("range", (0.0, 0.0))
                if isinstance(valence_range_raw, (tuple, list)) and len(valence_range_raw) == 2:
                    valence_range = cast(Tuple[float, float], tuple(valence_range_raw))
                else:
                    valence_range = (0.0, 0.0)  # Default fallback

                logger.info(f"Found {count_val} terms with emotional valence in 'emotional' dimension.")
                logger.info(
                    f"  Average valence: {mean_valence:.2f} (Range: [{valence_range[0]:.2f}, {valence_range[1]:.2f}])"
                )

                top_positive = cast(List[Tuple[str, float]], valence_analysis.get("top_positive", []))
                if top_positive:
                    logger.info("  Most positive terms:")
                    for term, val in top_positive:
                        logger.info(f"    - {term}: {val:.2f}")

                top_negative = cast(List[Tuple[str, float]], valence_analysis.get("top_negative", []))
                if top_negative:
                    logger.info("  Most negative terms:")
                    for term, val in top_negative:
                        logger.info(f"    - {term}: {val:.2f}")
            else:
                logger.info("No significant emotional valence data found in 'emotional' dimension.")

        except GraphError as ge:  # Catch specific graph errors like missing numpy
            logger.warning(f"Valence analysis skipped: {ge}")
        except Exception as e:
            logger.error(f"Unexpected error during valence analysis: {e}", exc_info=True)

        # --- Phase 4: Meta-Emotional Patterns ---
        logger.info("--- PHASE 4: META-EMOTIONAL PATTERNS ---")
        # Add sample meta-emotional data if needed
        logger.info("Ensuring sample meta-emotional data exists...")
        added_meta = False
        # Check if specific meta patterns exist before adding more
        meta_patterns_check = graph_manager.extract_meta_emotional_patterns()
        if not any(p for p in meta_patterns_check if p == "anxiety"):  # Example check
            sample_meta_relations = [
                ("anxiety", "fear", "meta_emotion", 0.8),
                ("regret", "sadness", "evokes", 0.7),
            ]
            for source, target, rel_type, weight in sample_meta_relations:
                try:
                    source_id = graph_manager.add_word_node(source, attributes={"valence": -0.7})
                    target_id = graph_manager.add_word_node(target, attributes={"valence": -0.8})
                    graph_manager.add_relationship(
                        source_id,
                        target_id,
                        relationship=rel_type,
                        dimension="emotional",
                        weight=weight,
                        color="#8a2be2",  # BlueViolet
                    )
                    added_meta = True
                except Exception as add_err:
                    logger.warning(f"Could not add sample meta-emotional relation {source}-{target}: {add_err}")
            if added_meta:
                logger.info("Added sample meta-emotional relationships.")

        meta_patterns = graph_manager.extract_meta_emotional_patterns()
        if meta_patterns:
            logger.info(f"Found {len(meta_patterns)} source terms in meta-emotional patterns.")
            logger.info("Sample meta-emotional patterns:")
            for source, targets in list(meta_patterns.items())[:3]:  # Show first 3
                target_str = ", ".join([f"{t['term']} ({t['relationship']})" for t in targets[:2]])
                logger.info(f"  - {source} → {target_str}...")
        else:
            logger.info("No meta-emotional patterns found.")

        # --- Phase 5: Emotional Transitions ---
        logger.info("--- PHASE 5: EMOTIONAL TRANSITIONS ---")
        try:
            transitions = graph_manager.analyze_emotional_transitions(
                path_length=2, min_transition_strength=0.05
            )  # Lower threshold for demo
            if transitions:
                logger.info(f"Found {len(transitions)} emotional transition pathways.")
                logger.info("Top emotional transitions (strength > 0.05, length <= 2):")
                for t in transitions[:3]:  # Show top 3
                    path_str = " → ".join(t["path"])
                    logger.info(f"  - Path: {path_str}")
                    logger.info(f"    Strength: {t['strength']:.3f}, Valence Shift: {t['valence_shift']:.2f}")
            else:
                logger.info("No significant emotional transitions found meeting criteria.")
        except Exception as e:
            logger.error(f"Error analyzing emotional transitions: {e}", exc_info=True)

        # --- Phase 6: Semantic Clusters ---
        logger.info("--- PHASE 6: SEMANTIC CLUSTERS ---")
        try:
            logger.info("Identifying semantic clusters (min size 2)...")
            # Use default weight 'weight', ensure nodes exist
            clusters = graph_manager.analyze_semantic_clusters(min_community_size=2)

            if clusters:
                logger.info(f"Found {len(clusters)} semantic clusters.")
                logger.info("Sample clusters:")
                for cluster_id, nodes_info in list(clusters.items())[:3]:  # Show first 3 clusters
                    logger.info(f"  Cluster {cluster_id}:")
                    for node_info in nodes_info[:3]:  # Show first 3 terms per cluster
                        term = node_info["term"]
                        valence = node_info.get("valence")
                        valence_str = f", valence: {valence:.2f}" if isinstance(valence, float) else ""
                        logger.info(f"    - {term}{valence_str}")
            else:
                logger.info("No significant semantic clusters found (min size 2).")
        except GraphAnalysisError as ga_err:  # Catch specific error for missing library
            logger.warning(f"Semantic clustering skipped: {ga_err}")
        except Exception as e:
            logger.error(f"Error during semantic clustering: {e}", exc_info=True)

        # --- Phase 7: Context Integration ---
        logger.info("--- PHASE 7: CONTEXT INTEGRATION ---")
        logger.info("Integrating emotional contexts (placeholder)...")
        # Contexts are stored but don't modify graph in current analysis implementation
        clinical_context = {"anger": 0.2, "fear": 0.5, "joy": 0.1}  # Example weights
        literary_context = {"anger": 1.5, "joy": 1.2, "sadness": 1.1}

        try:
            updated_clinical = graph_manager.integrate_emotional_context("clinical", clinical_context)
            updated_literary = graph_manager.integrate_emotional_context("literary", literary_context)
            logger.info(f"Stored 'clinical' context (affected {updated_clinical} elements - placeholder).")
            logger.info(f"Stored 'literary' context (affected {updated_literary} elements - placeholder).")

            # Demonstrate getting emotional subgraph (even without context application)
            context_term = "fear"  # Term likely involved in emotional edges
            logger.info(f"Extracting emotional subgraph around '{context_term}' (depth 1)...")
            emotional_subgraph: nx.Graph = graph_manager.get_emotional_subgraph(
                context_term, depth=1, min_intensity=0.1  # Use a threshold
            )
            logger.info(
                f"Extracted emotional subgraph: {emotional_subgraph.number_of_nodes()} nodes, {emotional_subgraph.number_of_edges()} edges."
            )

        except NodeNotFoundError as nnf:
            logger.warning(f"Could not extract emotional subgraph for '{context_term}': {nnf}")
        except Exception as e:
            logger.error(f"Error during context integration phase: {e}", exc_info=True)

        # --- Phase 8: Visualization ---
        logger.info("--- PHASE 8: VISUALIZATION ---")
        vis_paths: Dict[str, Path] = {
            "2d": output_dir / "graph_visualization_2d.html",
            "3d": output_dir / "graph_visualization_3d.html",
            "emotional": output_dir / "graph_visualization_emotional.html",
            "lexical": output_dir / "graph_visualization_lexical.html",
        }

        try:
            logger.info("Generating 2D interactive visualization (default dimensions)...")
            graph_manager.visualize_2d(output_path=str(vis_paths["2d"]), open_in_browser=False)  # Specify 2D

            logger.info("Generating 3D interactive visualization (default dimensions)...")
            graph_manager.visualize_3d(output_path=str(vis_paths["3d"]), open_in_browser=False)  # Specify 3D

            # Visualize specific dimensions if they exist
            if "emotional" in dimensions_data:
                logger.info("Generating visualization for 'emotional' dimension only...")
                graph_manager.visualize(  # Use default (likely 2D)
                    output_path=str(vis_paths["emotional"]),
                    dimensions_filter=["emotional"],
                    open_in_browser=False,
                )
            if "lexical" in dimensions_data:
                logger.info("Generating visualization for 'lexical' dimension only...")
                graph_manager.visualize(  # Use default (likely 2D)
                    output_path=str(vis_paths["lexical"]),
                    dimensions_filter=["lexical"],
                    open_in_browser=False,
                )

            logger.info("Visualizations saved to:")
            for key, path in vis_paths.items():
                if path.exists():
                    logger.info(f"  - {key.capitalize()}: {path}")
            logger.info("Open HTML files in a browser to explore.")

        except GraphVisualizationError as gv_err:  # Catch specific viz errors
            logger.warning(f"Visualization generation issue: {gv_err}")
        except ImportError as ie:  # Catch missing optional dependencies
            logger.warning(f"Visualization skipped due to missing library: {ie}")
        except Exception as e:
            logger.error(f"Unexpected error during visualization: {e}", exc_info=True)

        # --- Phase 9: Export ---
        logger.info("--- PHASE 9: EXPORT ---")
        gexf_path = output_dir / "lexical_graph_demo.gexf"
        logger.info(f"Saving complete graph to GEXF: {gexf_path}")
        try:
            graph_manager.save_to_gexf(str(gexf_path))
            logger.info(f"Graph saved successfully to {gexf_path}")
        except GraphError as ge:
            logger.error(f"Failed to save graph to GEXF: {ge}")
        except Exception as e:
            logger.error(f"Unexpected error saving GEXF: {e}", exc_info=True)

        # Export subgraph example
        try:
            logger.info(f"Exporting subgraph for '{example_term}' (depth 2)...")
            subgraph_path_str = graph_manager.export_subgraph(
                example_term,
                depth=2,
                output_path=str(output_dir / f"subgraph_{example_term}_depth2.gexf"),
            )
            if subgraph_path_str:
                logger.info(f"Subgraph exported successfully to {subgraph_path_str}")
            else:
                logger.warning(
                    f"Subgraph export for '{example_term}' resulted in an empty path (likely empty subgraph)."
                )
        except NodeNotFoundError:
            logger.warning(f"Could not extract subgraph for '{example_term}' (term not found).")
        except GraphError as ge:
            logger.error(f"Failed to export subgraph: {ge}")
        except Exception as e:
            logger.error(f"Unexpected error exporting subgraph: {e}", exc_info=True)

        elapsed_time = time.time() - start_time
        logger.info(f"Demonstration completed in {elapsed_time:.2f} seconds.")

    except GraphError as e:
        logger.error(f"A graph operation failed: {e}", exc_info=True)
        print(f"\nDemo aborted due to GraphError: {e}")
    except NodeNotFoundError as e:
        logger.error(f"Required node not found: {e}", exc_info=True)
        print(f"\nDemo aborted due to NodeNotFoundError: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the demo: {e}", exc_info=True)
        print(f"\nDemo aborted due to unexpected error: {e}")
        traceback.print_exc()
    finally:
        # Ensure DB connection is closed
        if db_manager:
            db_manager.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    graph_demo()
