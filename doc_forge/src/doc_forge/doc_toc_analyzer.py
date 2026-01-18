#!/usr/bin/env python3
# 🌀 Eidosian TOC Analysis System
"""
TOC Analysis - Structure & Flow Analysis for Documentation

This module analyzes table of contents structures to ensure perfect
navigation flow and hierarchical organization of documentation.
It follows Eidosian principles of structure, flow, and precision.
"""

import re
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

# Import project-wide utilities
from .utils.paths import get_repo_root, get_docs_dir
from .source_discovery import DocumentationDiscovery, discover_documentation

# 📊 Self-aware logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("doc_forge.toc_analyzer")

class TocAnalyzer:
    """
    Table of Contents Analyzer with Eidosian precision and insight.
    
    Analyzes documentation structure to provide metrics and recommendations.
    """
    
    def __init__(self, docs_dir: Path):
        """
        Initialize the TOC analyzer.
        
        Args:
            docs_dir: Documentation directory
        """
        self.docs_dir = docs_dir
        self.discovery = DocumentationDiscovery(docs_dir=docs_dir)
        self.documents = self.discovery.discover_all()
        self.toc_structure = self.discovery.generate_toc_structure(self.documents)
        self.metrics: Dict[str, Any] = {}
        self.recommendations: List[str] = []
        
    def analyze_toc_structure(self) -> Dict[str, Any]:
        """
        Analyze the TOC structure with Eidosian precision.
        
        Returns:
            Analysis results with metrics and recommendations
        """
        logger.info(f"🔍 Analyzing TOC structure for {self.docs_dir}")
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Combine metrics and recommendations into a complete analysis
        analysis = {
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "structure_quality": self._evaluate_structure_quality()
        }
        
        logger.info(f"✅ TOC structure analysis complete")
        return analysis
    
    def _calculate_metrics(self) -> None:
        """Calculate metrics for the TOC structure."""
        # Initialize metrics
        metrics = {
            "total_sections": len(self.toc_structure),
            "total_documents": 0,
            "section_sizes": {},
            "section_depths": {},
            "orphaned_documents": len(self.discovery.orphaned_documents),
            "balance_score": 0.0,
            "coverage_score": 0.0
        }
        
        # Calculate section-specific metrics
        section_sizes = {}
        section_proportions = {}
        max_size = 0
        
        for section_name, section_data in self.toc_structure.items():
            section_size = len(section_data["items"])
            section_sizes[section_name] = section_size
            metrics["total_documents"] += section_size
            
            if section_size > max_size:
                max_size = section_size
        
        # Calculate balance score (how evenly distributed content is)
        if metrics["total_sections"] > 0 and max_size > 0:
            ideal_size = metrics["total_documents"] / metrics["total_sections"]
            variance = sum((size - ideal_size) ** 2 for size in section_sizes.values()) / metrics["total_sections"]
            metrics["balance_score"] = 100 * (1 - (variance / (ideal_size ** 2 + 1)))  # +1 to avoid division by zero
            metrics["balance_score"] = max(0, min(100, metrics["balance_score"]))  # Clamp to 0-100
        
        # Calculate coverage score (percentage of documents in TOC vs. all discovered)
        all_discovered = sum(len(docs) for docs in self.documents.values())
        if all_discovered > 0:
            metrics["coverage_score"] = 100 * metrics["total_documents"] / all_discovered
        
        # Store metrics
        metrics["section_sizes"] = section_sizes
        self.metrics = metrics
    
    def _generate_recommendations(self) -> None:
        """Generate recommendations for TOC improvement."""
        recommendations = []
        
        # Check balance between sections
        unbalanced_threshold = 30  # Sections more than 30% larger than ideal
        ideal_size = self.metrics["total_documents"] / max(1, self.metrics["total_sections"])
        
        for section, size in self.metrics["section_sizes"].items():
            if size > (ideal_size * 1.3) and size > 5:  # Section is 30% larger than ideal and has >5 items
                recommendations.append(f"Consider splitting '{section}' section as it contains {size} items (ideal: {ideal_size:.1f})")
            elif size == 0:
                recommendations.append(f"Section '{section}' is empty - consider removing it or adding content")
                
        # Check for orphaned documents
        if self.metrics["orphaned_documents"] > 0:
            recommendations.append(f"Found {self.metrics['orphaned_documents']} orphaned documents - consider adding them to the TOC")
        
        # Balance score recommendations
        if self.metrics["balance_score"] < 70:
            recommendations.append("TOC structure is significantly unbalanced - consider redistributing content more evenly")
        
        # Check for missing essential sections
        essential_sections = {"getting_started", "reference"}
        existing_sections = set(self.toc_structure.keys())
        missing = essential_sections - existing_sections
        if missing:
            recommendations.append(f"Missing essential section(s): {', '.join(missing)}")
        
        self.recommendations = recommendations
    
    def _evaluate_structure_quality(self) -> Dict[str, Any]:
        """Evaluate the overall quality of the TOC structure."""
        # Calculate an overall quality score based on metrics
        balance_weight = 0.4  # Balance between sections: 40%
        coverage_weight = 0.4  # Coverage of documents: 40%
        orphan_weight = 0.2    # Orphaned documents penalty: 20%
        
        # Balance score is already calculated
        balance_score = self.metrics["balance_score"]
        
        # Coverage score is already calculated
        coverage_score = self.metrics["coverage_score"]
        
        # Orphan penalty
        orphan_score = 100
        if self.metrics["total_documents"] > 0:
            orphan_ratio = self.metrics["orphaned_documents"] / (self.metrics["total_documents"] + self.metrics["orphaned_documents"])
            orphan_score = 100 * (1 - orphan_ratio)
        
        # Calculate overall score
        overall_score = (balance_score * balance_weight) + (coverage_score * coverage_weight) + (orphan_score * orphan_weight)
        
        # Determine quality rating
        quality_rating = "Excellent"
        if overall_score < 60:
            quality_rating = "Poor"
        elif overall_score < 75:
            quality_rating = "Fair"
        elif overall_score < 85:
            quality_rating = "Good"
        elif overall_score < 95:
            quality_rating = "Very Good"
            
        return {
            "overall_score": overall_score,
            "quality_rating": quality_rating,
            "component_scores": {
                "balance_score": balance_score,
                "coverage_score": coverage_score,
                "orphan_score": orphan_score
            }
        }
    
    def visualize_structure(self, output_path: Optional[Path] = None) -> Path:
        """
        Generate a visual representation of the TOC structure.
        
        Args:
            output_path: Path to save the visualization (defaults to docs_dir/toc_visualization.html)
            
        Returns:
            Path to the generated visualization
        """
        if output_path is None:
            output_path = self.docs_dir / "toc_visualization.html"
        
        # Convert TOC structure to a format suitable for visualization
        visualization_data = {
            "name": "Documentation",
            "children": []
        }
        
        for section_name, section_data in self.toc_structure.items():
            section_node = {
                "name": section_data["title"],
                "children": []
            }
            
            for item in section_data["items"]:
                section_node["children"].append({
                    "name": item["title"],
                    "url": item["url"],
                    "value": 1  # Each document has equal weight
                })
            
            visualization_data["children"].append(section_node)
        
        # Add orphaned documents if any
        if self.discovery.orphaned_documents:
            orphan_node = {
                "name": "Orphaned Documents",
                "children": []
            }
            
            for orphan in self.discovery.orphaned_documents:
                orphan_node["children"].append({
                    "name": orphan.stem,
                    "url": str(orphan.relative_to(self.docs_dir)),
                    "value": 1
                })
            
            visualization_data["children"].append(orphan_node)
        
        # Create HTML with D3.js visualization
        html_content = self._generate_visualization_html(visualization_data)
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"📊 TOC visualization saved to {output_path}")
        return output_path
    
    def _generate_visualization_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML content for visualization."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation TOC Structure Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        #visualization {{
            width: 100%;
            height: 800px;
            border: 1px solid #ddd;
        }}
        .node circle {{
            fill: #fff;
            stroke: steelblue;
            stroke-width: 3px;
        }}
        .node text {{
            font: 12px sans-serif;
        }}
        .link {{
            fill: none;
            stroke: #ccc;
            stroke-width: 2px;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .metrics {{
            margin-top: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>Documentation Structure Visualization</h1>
    <div id="visualization"></div>
    <div class="metrics">
        <h2>Structure Metrics</h2>
        <p><strong>Total Sections:</strong> {self.metrics.get('total_sections', 0)}</p>
        <p><strong>Total Documents:</strong> {self.metrics.get('total_documents', 0)}</p>
        <p><strong>Orphaned Documents:</strong> {self.metrics.get('orphaned_documents', 0)}</p>
        <p><strong>Balance Score:</strong> {self.metrics.get('balance_score', 0):.2f}/100</p>
        <p><strong>Coverage Score:</strong> {self.metrics.get('coverage_score', 0):.2f}%</p>
        <p><strong>Overall Quality:</strong> {self._evaluate_structure_quality().get('quality_rating', 'Unknown')}</p>
    </div>
    
    <script>
        // Visualization data
        const treeData = {json.dumps(data)};
        
        // Set up dimensions and margins
        const margin = {{top: 20, right: 90, bottom: 30, left: 90}},
              width = 960 - margin.left - margin.right,
              height = 700 - margin.top - margin.bottom;
              
        // Create SVG
        const svg = d3.select("#visualization").append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
          .append("g")
            .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
            
        // Create tree layout
        const tree = d3.tree().size([height, width]);
        
        // Assigns parent, children, height, depth
        const root = d3.hierarchy(treeData, d => d.children);
        root.x0 = height / 2;
        root.y0 = 0;
        
        update(root);
        
        function update(source) {{
            // Assigns the x and y position for the nodes
            const treeData = tree(root);
            
            // Compute the new tree layout
            const nodes = treeData.descendants();
            const links = treeData.descendants().slice(1);
            
            // Normalize for fixed-depth
            nodes.forEach(d => {{ d.y = d.depth * 180 }});
            
            // ****************** Nodes section ***************************
            
            // Update the nodes...
            const node = svg.selectAll('g.node')
                .data(nodes, d => d.id || (d.id = ++i));
                
            // Enter any new nodes at the parent's previous position
            const nodeEnter = node.enter().append('g')
                .attr('class', 'node')
                .attr("transform", d => `translate(${{source.y0}},${{source.x0}})`)
                .on('click', click);
                
            // Add Circle for the nodes
            nodeEnter.append('circle')
                .attr('r', 10)
                .style("fill", d => d._children ? "lightsteelblue" : "#fff");
                
            // Add labels for the nodes
            nodeEnter.append('text')
                .attr("dy", ".35em")
                .attr("x", d => d.children || d._children ? -13 : 13)
                .attr("text-anchor", d => d.children || d._children ? "end" : "start")
                .text(d => d.data.name);
                
            // UPDATE
            const nodeUpdate = nodeEnter.merge(node);
            
            // Transition to the proper position for the node
            nodeUpdate.transition()
                .duration(750)
                .attr("transform", d => `translate(${{d.y}},${{d.x}})`);
                
            // Update the node attributes and style
            nodeUpdate.select('circle')
                .attr('r', 10)
                .style("fill", d => d._children ? "lightsteelblue" : "#fff")
                .attr('cursor', 'pointer');
                
            // Remove any exiting nodes
            const nodeExit = node.exit().transition()
                .duration(750)
                .attr("transform", d => `translate(${{source.y}},${{source.x}})`)
                .remove();
                
            // On exit reduce the node circles size to 0
            nodeExit.select('circle')
                .attr('r', 1e-6);
                
            // On exit reduce the opacity of text labels
            nodeExit.select('text')
                .style('fill-opacity', 1e-6);
                
            // ****************** links section ***************************
            
            // Update the links...
            const link = svg.selectAll('path.link')
                .data(links, d => d.id);
                
            // Enter any new links at the parent's previous position
            const linkEnter = link.enter().insert('path', "g")
                .attr("class", "link")
                .attr('d', d => {{
                    const o = {{x: source.x0, y: source.y0}};
                    return diagonal(o, o);
                }});
                
            // UPDATE
            const linkUpdate = linkEnter.merge(link);
            
            // Transition back to the parent element position
            linkUpdate.transition()
                .duration(750)
                .attr('d', d => diagonal(d, d.parent));
                
            // Remove any exiting links
            link.exit().transition()
                .duration(750)
                .attr('d', d => {{
                    const o = {{x: source.x, y: source.y}};
                    return diagonal(o, o);
                }})
                .remove();
                
            // Store the old positions for transition
            nodes.forEach(d => {{
                d.x0 = d.x;
                d.y0 = d.y;
            }});
            
            // Creates a curved (diagonal) path from parent to the child nodes
            function diagonal(s, d) {{
                return `M ${{s.y}} ${{s.x}}
                        C ${{(s.y + d.y) / 2}} ${{s.x}},
                          ${{(s.y + d.y) / 2}} ${{d.x}},
                          ${{d.y}} ${{d.x}}`;
            }}
            
            // Toggle children on click
            function click(event, d) {{
                if (d.children) {{
                    d._children = d.children;
                    d.children = null;
                }} else {{
                    d.children = d._children;
                    d._children = null;
                }}
                update(d);
            }}
        }}
        
        // Variable to track node IDs
        let i = 0;
    </script>
</body>
</html>
"""


def analyze_toc(docs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Analyze the TOC structure of documentation with Eidosian precision.
    
    This function serves as the universal interface to the TOC analysis system,
    providing insights into the documentation structure quality and balance.
    
    Args:
        docs_dir: Documentation directory (auto-detected if None)
        
    Returns:
        Analysis results with metrics, recommendations and quality assessment
    """
    # Auto-detect docs directory if not provided
    if docs_dir is None:
        try:
            docs_dir = get_docs_dir()
        except Exception as e:
            logger.error(f"❌ Failed to auto-detect docs directory: {e}")
            return {"error": str(e)}
    
    docs_dir = Path(docs_dir)
    
    if not docs_dir.is_dir():
        logger.error(f"❌ Documentation directory not found or not a directory: {docs_dir}")
        return {"error": f"Documentation directory not found: {docs_dir}"}
    
    try:
        # Create analyzer and run analysis
        analyzer = TocAnalyzer(docs_dir)
        results = analyzer.analyze_toc_structure()
        
        # Generate visualization
        visualization_path = analyzer.visualize_structure()
        results["visualization"] = str(visualization_path)
        
        logger.info(f"✅ TOC analysis complete. Quality: {results['structure_quality']['quality_rating']}")
        return results
    except Exception as e:
        logger.error(f"❌ TOC analysis failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze documentation TOC structure.")
    parser.add_argument("docs_dir", nargs="?", type=Path, help="Documentation directory")
    parser.add_argument("--output", "-o", type=Path, help="Output file for analysis results")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()
    
    results = analyze_toc(args.docs_dir)
    
    if args.json or args.output:
        output_data = json.dumps(results, indent=2)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_data)
            print(f"✅ Analysis results saved to {args.output}")
        else:
            print(output_data)
    else:
        print("\n📊 TOC Structure Analysis Results:")
        print(f"Quality Rating: {results['structure_quality']['quality_rating']}")
        print(f"Overall Score: {results['structure_quality']['overall_score']:.2f}/100")
        
        print("\n📏 Metrics:")
        for key, value in results["metrics"].items():
            if key != "section_sizes":
                print(f"  {key}: {value}")
        
        print("\nSection Sizes:")
        for section, size in results["metrics"]["section_sizes"].items():
            print(f"  {section}: {size}")
        
        print("\n💡 Recommendations:")
        if results["recommendations"]:
            for rec in results["recommendations"]:
                print(f"  • {rec}")
        else:
            print("  No recommendations - structure looks good!")
        
        print(f"\n🎨 Visualization saved to: {results['visualization']}")