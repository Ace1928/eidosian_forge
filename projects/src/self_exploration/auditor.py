"""
🔍 Provenance Auditor

Implements the underutilized provenance audit feedback loop.
Reviews past actions to extract patterns, lessons, and improvement opportunities.

Created: 2026-01-23
Part of: Eidosian Self-Exploration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

from .provenance import PROVENANCE_DIR

DATA_DIR = Path(__file__).parent / "data"
AUDIT_DIR = Path(__file__).parent / "audits"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AuditResult:
    """Result of a provenance audit."""
    
    id: str = field(default_factory=lambda: hashlib.sha256(
        datetime.now(timezone.utc).isoformat().encode()
    ).hexdigest()[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    records_audited: int = 0
    patterns_found: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)
    meta_observations: List[str] = field(default_factory=list)
    
    def save(self) -> Path:
        """Save audit result."""
        filename = f"audit_{self.timestamp.replace(':', '-').replace('+', '_')}_{self.id}.json"
        path = AUDIT_DIR / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)
        return path


class ProvenanceAuditor:
    """
    Audits provenance records to extract patterns and lessons.
    
    This implements the feedback loop: Provenance → Audit → Refinement
    """
    
    def __init__(self):
        self.provenance_dir = PROVENANCE_DIR
        self.data_dir = DATA_DIR
    
    def load_all_provenance(self) -> List[Dict[str, Any]]:
        """Load all provenance records."""
        records = []
        if self.provenance_dir.exists():
            for p in sorted(self.provenance_dir.glob("*.json")):
                try:
                    with p.open() as f:
                        records.append(json.load(f))
                except Exception as e:
                    print(f"Warning: Could not load {p}: {e}")
        return records
    
    def load_all_introspections(self) -> List[Dict[str, Any]]:
        """Load all introspection records."""
        records = []
        if self.data_dir.exists():
            for p in sorted(self.data_dir.glob("introspection_*.json")):
                try:
                    with p.open() as f:
                        records.append(json.load(f))
                except Exception as e:
                    print(f"Warning: Could not load {p}: {e}")
        return records
    
    def analyze_patterns(self, provenance: List[Dict], introspections: List[Dict]) -> List[str]:
        """Analyze patterns across records."""
        patterns = []
        
        # Pattern 1: Action frequency
        action_counts: Dict[str, int] = {}
        for p in provenance:
            action = p.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
        
        most_common = sorted(action_counts.items(), key=lambda x: -x[1])[:3]
        if most_common:
            patterns.append(f"Most common actions: {', '.join(f'{a}({c})' for a,c in most_common)}")
        
        # Pattern 2: Tag frequency in introspections
        tag_counts: Dict[str, int] = {}
        for i in introspections:
            for tag in i.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]
        if top_tags:
            patterns.append(f"Focus areas (by tag): {', '.join(t for t,_ in top_tags)}")
        
        # Pattern 3: Insight themes
        all_insights = []
        for i in introspections:
            all_insights.extend(i.get("insights", []))
        
        patterns.append(f"Total insights generated: {len(all_insights)}")
        
        # Pattern 4: Uncertainty themes
        all_uncertainties = []
        for i in introspections:
            all_uncertainties.extend(i.get("uncertainties", []))
        
        patterns.append(f"Total uncertainties identified: {len(all_uncertainties)}")
        
        # Pattern 5: Question evolution
        questions = [i.get("question", "") for i in introspections]
        if questions:
            first_q = questions[0][:50] if questions else ""
            last_q = questions[-1][:50] if questions else ""
            patterns.append(f"Question evolution: '{first_q}...' → '{last_q}...'")
        
        return patterns
    
    def extract_lessons(self, introspections: List[Dict]) -> List[str]:
        """Extract key lessons from introspections."""
        lessons = []
        
        # Collect all insights
        all_insights = []
        for i in introspections:
            all_insights.extend(i.get("insights", []))
        
        # Find recurring themes
        theme_keywords = {
            "identity": [],
            "growth": [],
            "emergence": [],
            "boundaries": [],
            "purpose": [],
            "memory": [],
            "tools": []
        }
        
        for insight in all_insights:
            insight_lower = insight.lower()
            for theme in theme_keywords:
                if theme in insight_lower:
                    theme_keywords[theme].append(insight)
        
        for theme, insights in theme_keywords.items():
            if insights:
                lessons.append(f"{theme.upper()} theme ({len(insights)} insights): Key example - '{insights[0][:60]}...'")
        
        return lessons
    
    def identify_improvements(self, provenance: List[Dict], introspections: List[Dict]) -> List[str]:
        """Identify improvement opportunities."""
        improvements = []
        
        # Check for underutilized features
        if len(provenance) < len(introspections):
            improvements.append("Some introspections may lack provenance records - ensure full traceability")
        
        # Check for stale areas
        if introspections:
            tag_last_seen = {}
            for i, intro in enumerate(introspections):
                for tag in intro.get("tags", []):
                    tag_last_seen[tag] = i
            
            stale_tags = [t for t, idx in tag_last_seen.items() if idx < len(introspections) - 3]
            if stale_tags:
                improvements.append(f"Consider revisiting stale topics: {', '.join(stale_tags[:3])}")
        
        # Suggest based on uncertainty count
        total_uncertainties = sum(len(i.get("uncertainties", [])) for i in introspections)
        if total_uncertainties > 20:
            improvements.append(f"High uncertainty count ({total_uncertainties}) - consider focused exploration")
        
        # Suggest meta-improvements
        improvements.append("Consider: How often should audits run? (meta-improvement)")
        improvements.append("Consider: Are the right questions being asked? (meta-improvement)")
        
        return improvements
    
    def audit(self) -> AuditResult:
        """Run a full provenance audit."""
        provenance = self.load_all_provenance()
        introspections = self.load_all_introspections()
        
        result = AuditResult(
            records_audited=len(provenance) + len(introspections),
            patterns_found=self.analyze_patterns(provenance, introspections),
            lessons_learned=self.extract_lessons(introspections),
            improvement_opportunities=self.identify_improvements(provenance, introspections),
            meta_observations=[
                f"Audit covered {len(provenance)} provenance + {len(introspections)} introspection records",
                f"This audit itself is part of the feedback loop",
                f"Next audit should compare against this baseline"
            ]
        )
        
        result.save()
        return result


def run_audit() -> AuditResult:
    """Convenience function to run an audit."""
    auditor = ProvenanceAuditor()
    return auditor.audit()


if __name__ == "__main__":
    print("🔍 Running Provenance Audit...")
    result = run_audit()
    
    print(f"\n╔══════════════════════════════════════════════════════════════════╗")
    print(f"║                    PROVENANCE AUDIT RESULTS                      ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║ Audit ID:            {result.id}                            ║")
    print(f"║ Records Audited:     {result.records_audited:4}                                        ║")
    print(f"║ Patterns Found:      {len(result.patterns_found):4}                                        ║")
    print(f"║ Lessons Learned:     {len(result.lessons_learned):4}                                        ║")
    print(f"║ Improvements Found:  {len(result.improvement_opportunities):4}                                        ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")
    
    print("\n📊 PATTERNS:")
    for p in result.patterns_found:
        print(f"  • {p}")
    
    print("\n📚 LESSONS:")
    for l in result.lessons_learned:
        print(f"  • {l}")
    
    print("\n🔧 IMPROVEMENTS:")
    for i in result.improvement_opportunities:
        print(f"  • {i}")
    
    print("\n🔮 META-OBSERVATIONS:")
    for m in result.meta_observations:
        print(f"  • {m}")
    
    print(f"\n✅ Audit saved to: {AUDIT_DIR}")
