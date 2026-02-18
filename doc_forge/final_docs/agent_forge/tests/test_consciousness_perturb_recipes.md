Loading model... 


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8083-ae2d3f28a
model      : qwen2.5-1.5b-instruct-q5_k_m.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read               add a text file


> <|im_start|>system
You are an Eidosian Documentation Forge agent. Your task is to generate detailed, accurate, and structurally elegant documentation for the provided source file.
Follow these requirements:
1. File Summary: High-level overview of what the file does.
2. API Documentation: List all functions, classes, and their parameters/methods.
3. Current Status: Accurate assessment of the file's current state based on its content and comments.
4. Potential Future Directions: Eidosian ideas for ... (truncated)

# File Summary

`agent_forge/tests/test_consciousness_perturb_recipes.py` contains tests for various recipes in the `ConsciousnessPerturbRecipe` module.

## API Documentation

### Test Recipe Catalog Contains Expected Recipes

- **Function:** `test_recipe_catalog_contains_expected_recipes`
- **Parameters/Metrics:**
  - None
- **Return Value:** `None`

This test checks if the available recipes include expected recipes such as "sensory_deprivation", "attention_flood", etc.

### Test Recipe Expected Signature Evaluation

- **Function:** `test_recipe_expected_signature_evaluation`
- **Parameters/Metrics:**
  - `recipe`: PerturbationRecipe
- **Return Value:** `None`

This test evaluates the expected signatures of a given recipe for continuity_delta, perspective_coherence_delta, and coherence_delta.

### Test Apply Recipe Registers All Perturbations

- **Function:** `test_apply_recipe_registers_all_perturbations`
- **Parameters/Metrics:**
  - `stub`: _KernelStub
  - `recipe`: PerturbationRecipe
- **Return Value:** `None`

This test verifies that an apply recipe registers all perturbations.

### Test Bench Runner Expands Recipe Entries

- **Function:** `test_bench_runner_expands_recipe_entries`
- **Parameters/Metrics:**
  - `base`: Path
  - `runner`: ConsciousnessBenchRunner
  - `spec`: TrialSpec
- **Return Value:** `None`

This test runs a trial using the bench runner and checks if the report contains recipe entries for wm_lesion.

## Current Status

The file is well-commented and covers the following functionalities:

- Test Recipe Catalog Contains Expected Recipes: Ensures recipes are available.
- Test Recipe Expected Signature Evaluation: Evaluates expected signatures of a given recipe.
- Test Apply Recipe Registers All Perturbations: Verifies that apply recipe registers all perturbations.
- Test Bench Runner Expands Recipe Entries: Checks if bench runner expands recipe entries.

The code is clean and easy to understand, but it can be refactored for better readability and maintainability. For example:

```python
# Add docstrings at the top of each function for better clarity

def test_recipe_catalog_contains_expected_recipes():
    names = set(available_recipes())
    assert "sensory_deprivation" in names
    assert "attention_flood" in names
    assert "identity_wobble" in names
    assert "wm_lesion" in names
    assert "dopamine_spike" in names
    assert "gw_bottleneck_strain" in names
    assert "world_model_scramble" in names

def test_recipe_expected_signature_evaluation(recipe):
    expected = {
        "continuity_delta": -0.22,
        "perspective_coherence_delta": -0.08,
        "coherence_delta": -0.03,
    }
    eval_out = evaluate_expected_signatures(recipe.expected_signatures, expected)
    assert eval_out["defined"] is True
    assert eval_out["pass"] is True
    assert len(eval_out["checks"]) == len(recipe.expected_signatures)

def test_apply_recipe_registers_all_perturbations(stub, recipe):
    result = apply_recipe(stub, recipe)
    assert result.applied is True
    assert result.details.get("recipe") == "attention_flood"
    assert result.details.get("count") == len(recipe.perturbations)
    assert len(stub.payloads) == len(recipe.perturbations)

def test_bench_runner_expands_recipe_entries(spec, runner, base):
    report = runner.run_trial(spec, persist=False)
    recipes = report.get("recipes", [])
    expectations = report.get("recipe_expectations", {})
    
    if recipes:
        assert recipes[0].get("name") == "wm_lesion"
        assert expectations.get("defined") is True
```

## Potential Future Directions

- Consider adding more detailed test cases for different types of recipe evaluations.
- Improve the readability and maintainability by refactoring the code.

[ Prompt: 51.8 t/s | Generation: 6.4 t/s ]

Exiting...