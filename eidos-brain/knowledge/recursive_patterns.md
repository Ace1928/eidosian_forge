# Recursive Patterns

These patterns guide the use of templates in `templates.md` and define how
documentation references are organized in the `glossary_reference.md`.

## Memory Recursion Pattern
1. Collect experiences.
2. Reflect on stored memories.
3. Generate insights via meta-reflection.
4. Append insights as new memories for future cycles.

### Combined Cycle Pattern
Use `process_cycle` to store an experience and immediately recurse,
appending reflective insights in a single step.

**Steps**
1. Accept an experience from the caller.
2. Call ``EidosCore.process_cycle`` with the experience.
3. The method stores the experience via ``remember`` and then runs ``recurse``.
4. ``recurse`` uses :class:`MetaReflection` to generate insights.
5. Append the insights to memory for future cycles.

This pattern reduces boilerplate when collecting new experiences in a loop and
ensures every input is reflected upon exactly once.

## CLI Recursion Patterns

### Interactive Combined Cycle
1. Launch a command loop that prompts the user for experiences.
2. For each input, call ``process_cycle`` and print the generated insights.
3. Continue until the user exits.

### Batch Reflection Pattern
1. Accumulate a list of experiences from a script or file.
2. Load them into memory with ``remember`` or ``load_memory``.
3. Invoke ``recurse`` once to analyze the entire batch.
4. Optionally persist results for later review.
