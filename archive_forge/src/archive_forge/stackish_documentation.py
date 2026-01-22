from pyparsing import Suppress,Word,nums,alphas,alphanums,Combine,oneOf,\

NUMBER A simple integer type that's just any series of digits.
FLOAT A simple floating point type.
STRING A string is double quotes with anything inside that's not a " or
    newline character. You can include 
 and " to include these
    characters.
MARK Marks a point in the stack that demarcates the boundary for a nested
    group.
WORD Marks the root node of a group, with the other end being the nearest
    MARK.
GROUP Acts as the root node of an anonymous group.
ATTRIBUTE Assigns an attribute name to the previously processed node.
    This means that just about anything can be an attribute, unlike in XML.
BLOB A BLOB is unique to Stackish and allows you to record any content
    (even binary content) inside the structure. This is done by pre-
    sizing the data with the NUMBER similar to Dan Bernstein's netstrings
    setup.
SPACE White space is basically ignored. This is interesting because since
    Stackish is serialized consistently this means you can use 
 as the
    separation character and perform reasonable diffs on two structures.
