import pytest
from unittest.mock import MagicMock, patch
from word_forge.database.database_manager import DBManager
from word_forge.linguistics.phonetics_manager import PhoneticsManager
from word_forge.linguistics.g2p_manager import G2PManager, G2PResult
from word_forge.linguistics.prosody import ProsodyEngine
from word_forge.phrases.phrase_manager import PhraseManager
from word_forge.parser.parser_refiner import ParserRefiner

@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test_word_forge.db"
    db_mgr = DBManager(db_file)
    db_mgr.create_tables()
    yield db_mgr
    db_mgr.close()

def test_phonetics_manager(db):
    pm = PhoneticsManager(db)
    word_id = 1
    db.insert_or_update_word("test")
    
    # Test upsert
    p_id = pm.upsert_phonetics(word_id, "tɛst", arpabet="T EH1 S T", stress_pattern="1")
    assert p_id > 0
    
    # Test variants
    pm.add_variant(p_id, "tɛst_fast", context="fast-speech")
    
    # Test retrieval
    results = pm.get_phonetics(word_id)
    assert len(results) == 1
    assert results[0]["ipa"] == "tɛst"
    assert len(results[0]["variants"]) == 1
    assert results[0]["variants"][0]["ipa"] == "tɛst_fast"

def test_g2p_manager():
    # Test exceptions
    gm = G2PManager()
    res = gm.convert("eidos")
    assert res["ipa"] == "ˈaɪ.dɒs"
    assert res["stress_pattern"] == "1-0"
    
    # Test manual exception
    gm.add_exception("test", G2PResult(ipa="tɛst", arpabet="T EH1 S T", stress_pattern="1"))
    res = gm.convert("test")
    assert res["ipa"] == "tɛst"

def test_phrase_manager(db):
    pm = PhraseManager(db)
    
    # Create component words
    db.insert_or_update_word("hello")
    db.insert_or_update_word("world")
    w1 = db.get_word_id("hello")
    w2 = db.get_word_id("world")
    
    # Test upsert
    phrase_id = pm.upsert_phrase("hello world", prosody_priors={"pitch": 1.2})
    assert phrase_id > 0
    
    # Test components
    pm.set_phrase_components(phrase_id, [w1, w2])
    
    # Test retrieval
    phrase = pm.get_phrase("hello world")
    assert phrase["text"] == "hello world"
    assert phrase["prosody_priors"]["pitch"] == 1.2
    assert "hello world" in phrase["component_terms"]
    
    # Test realization
    pm.record_realization(phrase_id, "agent_1", realization_ipa="hɛloʊ wɜrld")
    
    # Test occurrence tracking
    pm.track_occurrence("hello world")
    updated = pm.get_phrase("hello world")
    assert updated["novelty_score"] < 1.0

def test_prosody_engine():
    pe = ProsodyEngine()
    priors = pe.generate_priors("test phrase", valence=0.8, arousal=0.9)
    assert priors["pitch_multiplier"] > 1.0
    assert priors["intensity_multiplier"] > 1.0
    assert priors["duration_multiplier"] < 1.0

@patch("word_forge.linguistics.g2p_manager.validated_query")
@patch("word_forge.linguistics.prosody.validated_query")
@patch("word_forge.parser.parser_refiner.validated_query")
@patch("word_forge.parser.lexical_functions.create_lexical_dataset")
def test_integrated_refiner(mock_create_dataset, mock_parser_vq, mock_prosody_vq, mock_g2p_vq, db, caplog):
    caplog.set_level("DEBUG")
    # Setup mock returns for all validated_query mocks
    for mock_vq in [mock_parser_vq, mock_prosody_vq, mock_g2p_vq]:
        mock_vq.return_value.is_success = True
        mock_vq.return_value.unwrap.return_value = {
            "ipa": "əˈmɛzɪŋ",
            "arpabet": "AH0 M EY1 Z IH0 NG",
            "stress_pattern": "0-1-0-0",
            "phrases": ["amazing grace", "amazing discovery"],
            "entities": [],
            "pitch_multiplier": 1.1,
            "duration_multiplier": 0.9,
            "intensity_multiplier": 1.2
        }
    
    # Mocking lexical dataset
    mock_create_dataset.return_value = {
        "word": "amazing",
        "wordnet_data": [{"definition": "causing great surprise", "part_of_speech": "adj"}],
        "openthesaurus_synonyms": ["incredible"],
        "odict_data": {},
        "dbnary_data": [],
        "opendict_data": {},
        "thesaurus_synonyms": [],
        "example_sentence": "It was an amazing sight."
    }
    
    refiner = ParserRefiner(db_manager=db)
    result = refiner.process_word("amazing")
    assert result.is_success
    
    # Wait for background tasks to complete
    refiner._executor.shutdown(wait=True)
    
    # Verify phonetics stored
    pm = PhoneticsManager(db)
    p = pm.get_phonetics_by_term("amazing")
    assert len(p) == 1
    assert p[0]["ipa"] == "əˈmɛzɪŋ"
    
    # Verify phrases discovered
    phm = PhraseManager(db)
    phrase = phm.get_phrase("amazing grace")
    assert phrase is not None
    assert phrase["prosody_priors"] is not None
