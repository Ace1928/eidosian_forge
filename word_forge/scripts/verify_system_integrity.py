import sys
import os
import logging
from pathlib import Path

# Setup paths
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("../lib"))

from word_forge.database.database_manager import DBManager
from word_forge.linguistics.g2p_manager import G2PManager
from word_forge.linguistics.morphology import MorphologyManager
from word_forge.linguistics.prosody import ProsodyEngine
from word_forge.phrases.phrase_manager import PhraseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integrity_check")

def verify_all():
    db_path = "data/integrity_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = DBManager(db_path)
    db.create_tables()
    
    print("\n--- 1. G2P Manager Check ---")
    g2p = G2PManager()
    res = g2p.convert("eidos")
    print(f"Eidos G2P: {res}")
    assert res["ipa"] == "ˈaɪ.dɒs"
    
    print("\n--- 2. Morphology Manager Check ---")
    morph = MorphologyManager(db)
    # Morfessor needs training for real results, but we check if it runs
    parts = morph.decompose("recursive")
    print(f"Recursive Decomposition: {parts}")
    assert len(parts) > 0
    
    m_id = morph.upsert_morpheme("re-", m_type="prefix")
    print(f"Upserted morpheme ID: {m_id}")
    
    print("\n--- 3. Prosody Engine Check ---")
    prosody = ProsodyEngine()
    priors = prosody.generate_priors("Hello world", valence=0.5, arousal=0.5)
    print(f"Prosody Priors: {priors}")
    assert "pitch_multiplier" in priors
    
    print("\n--- 4. Phrase Manager Check ---")
    phrase_mgr = PhraseManager(db)
    db.insert_or_update_word("hello")
    db.insert_or_update_word("world")
    w1 = db.get_word_id("hello")
    w2 = db.get_word_id("world")
    
    p_id = phrase_mgr.upsert_phrase("hello world", prosody_priors=priors)
    phrase_mgr.set_phrase_components(p_id, [w1, w2])
    
    phrase_data = phrase_mgr.get_phrase("hello world")
    print(f"Retrieved Phrase: {phrase_data}")
    assert phrase_data["text"] == "hello world"
    assert "hello" in phrase_data["component_terms"]
    
    print("\n--- INTEGRITY CHECK PASSED ---")
    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    try:
        verify_all()
    except Exception as e:
        logger.error(f"Integrity check failed: {e}", exc_info=True)
        sys.exit(1)
