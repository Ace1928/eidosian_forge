def maxCtxContextualRule(maxCtx, st, chain):
    """Calculate usMaxContext based on a contextual feature rule."""
    if not chain:
        return max(maxCtx, st.GlyphCount)
    elif chain == 'Reverse':
        return max(maxCtx, st.GlyphCount + st.LookAheadGlyphCount)
    return max(maxCtx, st.InputGlyphCount + st.LookAheadGlyphCount)