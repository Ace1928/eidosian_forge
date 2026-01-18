╔═══════════════════════════════════════════════════════════════════════════╗
║ ███████╗██╗ ██████╗ ███████╗ ██████╗ ███╗ ██╗████████╗ ║
║ ██╔════╝██║██╔════╝ ██╔════╝██╔═══██╗████╗ ██║╚══██╔══╝ ║
║ █████╗ ██║██║ ███╗█████╗ ██║ ██║██╔██╗ ██║ ██║ ║
║ ██╔══╝ ██║██║ ██║██╔══╝ ██║ ██║██║╚██╗██║ ██║ ║
║ ██║ ██║╚██████╔╝██║ ╚██████╔╝██║ ╚████║ ██║ ║
║ ╚═╝ ╚═╝ ╚═════╝ ╚═╝ ╚═════╝ ╚═╝ ╚═══╝ ╚═╝ ║
╚═══════════════════════════════════════════════════════════════════════════╝

           ⚜️ The FIGfont Version 2 FIGfont and FIGdriver Standard ⚜️
           ═══ ═══════ ═══════ ═ ═══════ ═══ ═════════ ════════
              Draft 2.0 Copyright 1996, 1997
              by John Cowan and Paul Burton
              Portions Copyright 1991, 1993, 1994
              by Glenn Chappell and Ian Chai
              May be freely copied and distributed.

              ⟡ Eidosian Edition 2025 ⟡
              Figlet Forge: The Eidosian Typography Engine

             📍 Original Figlet: http://www.figlet.org/

      ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
      ┃  _____          __           __                            ┃
      ┃ / ___/__  ___  / /____ ___  / /____                        ┃
      ┃/ /__/ _ \/ _ \/ __/ -_) _ \/ __(_-<                        ┃
      ┃\___/\___/_//_/\__/\__/_//_/\__/___/                        ┃
      ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

        ╭─────────────────────────────────────────────────────╮
        │ ◈ INTRODUCTION                                      │
        │ ◈ BASIC DEFINITIONS AND CONCEPTS                    │
        │   • "FIGfont"                                       │
        │   • "FIGcharacters" and "Sub-characters"            │
        │   • "FIGdriver"                                     │
        │   • "FIGure"                                        │
        │   • "FIG"                                           │
        │   • "Layout Modes"                                  │
        │   • "Smushing Rules"                                │
        │   • "Hardblanks"                                    │
        │ ◈ CREATING FIGFONTS                                 │
        │   • The Header Line                                 │
        │   • Interpretation of Layout Parameters             │
        │   • Setting Layout Parameters Step-by-Step          │
        │   • FIGfont Comments                                │
        │   • FIGcharacter Data                               │
        │     - Basic Data Structure                          │
        │     - Character Codes                               │
        │     - Required FIGcharacters                        │
        │     - Code Tagged FIGcharacters                     │
        │ ◈ NOTES - AVOIDING ERRORS AND GENERAL ADVICE        │
        │ ◈ CONTROL FILES                                     │
        │   • Standard Format                                 │
        │   • Extended Commands                               │
        │ ◈ STANDARDIZED CAPABILITIES OF CURRENT AND FUTURE   │
        │   FIGDRIVERS                                        │
        │ ◈ CHART OF CAPABILITIES OF FIGLET 2.2.1 AND         │
        │   FIGWIN 1.0                                        │
        ╰─────────────────────────────────────────────────────╯

# 🔍 INTRODUCTION

This document specifies the format of font files and associated control files used by
FIGlet and FIGWin programs (collectively known as FIGdrivers). It serves as a
comprehensive reference for designers creating fonts (FIGfonts) compatible with these
systems and establishes standards for future FIGdriver development.

┌─────────────────────────────────────────────────────────────────────┐
│ 📝 PURPOSE: │
│ • Define standard font file format specifications │
│ • Document control file structures and behaviors │
│ • Establish compatibility guidelines across implementations │
│ • Serve as canonical reference for font designers │
└─────────────────────────────────────────────────────────────────────┘

Note that some features described in this document may not be supported by all
FIGdriver implementations. Consult program-specific documentation for details on
using FIGlet or FIGWin directly.

> ⚠️ **IMPLEMENTATION NOTE**: FIGWin 1.0 includes a program called FIGfont Editor
> for Windows 1.0 that simplifies font creation without requiring complete
> understanding of this specification. However, familiarity with the "BASIC
> DEFINITIONS AND CONCEPTS" section is still recommended for effective use.

## 📤 CONTRIBUTING FONTS

If you design a FIGfont, please share it with the community:

1. Send an announcement email to the FIGlet fonts mailing list:
   <figletfonts@figlet.org>

2. Submit a copy to <info@figlet.org> for potential inclusion in the
   official repository at ftp://ftp.figlet.org/

🔄 Contributing your work helps maintain a vibrant ecosystem of ASCII typography
resources for all users.

# BASIC DEFINITIONS AND CONCEPTS

## "FIGfont" 🔤

A FIGfont is a file which represents the graphical arrangement of characters representing larger characters. Since a FIGfont file is a text file, it can be created with any text editing program on any platform. The filename of a FIGfont file must end with ".flf", which stands for "<F>IG<L>ettering <F>ont".

```ascii
┌───────────────────────────────────────────────────────────┐
│ 📂 FIGfont = Typography patterns crystallized into text  │
└───────────────────────────────────────────────────────────┘
```

## "FIGcharacters" and "Sub-characters" 📝

Because FIGfonts describe large characters which consist of smaller characters, confusion can result when discussing one or the other. Therefore, the terms "FIGcharacter" and "sub-character" are used, respectively.

```ascii
╭───────────────────────────────────────────────────────────╮
│ ╭───╮  ← This entire structure is a FIGcharacter         │
│ │ A │  ← These individual characters are sub-characters  │
│ ╰───╯                                                    │
╰───────────────────────────────────────────────────────────╯
```

## "FIGdriver" ⚙️

The term FIGdriver is used in this document to encompass FIGlet, FIGWin, and any future programs which use FIGfonts.

## "FIGure" 🖼️

A FIGure (thusly capitalized) is an image created by a FIGdriver.

## "FIG" 📜 Evolution History

> _"Understanding where we've been illuminates the path ahead."_ — Eidosian Principle 4.7

A bit of history:

In Spring 1991, inspired by the Email signature of a friend named Frank, and goaded on by Ian Chai, Glenn Chappell wrote a nifty little 170-line "C" program called "newban", which would create large letters out of ordinary text characters. At the time, it was only compiled for UNIX. In hindsight, we now call it "FIGlet 1.0". FIGlet stands for <F>rank, <I>an, and <G>lenn's <let>ters. In various incarnations, newban circulated around the net for a couple of years. It had one font, which included only lowercase letters.

```ascii
┌──────────────────────────────────────────────────────────┐
│ 🌱 EVOLUTIONARY TIMELINE                                │
│                                                        │
│ ┌────────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐  │
│ │ newban 1.0 │→ │FIGlet 2.0│→│FIGlet 2.1│→│FIGlet 2.2│  │
│ │ 170 lines  │  │ 888 lines│  │1314 lines│  │ 4800 lines│  │
│ └────────────┘  └─────────┘  └──────────┘  └────────┘  │
│      1991         1993         1994-1995      1996      │
└──────────────────────────────────────────────────────────┘
```

In early 1993, Ian decided newban was due for a few changes, so together Ian and Glenn added the full ASCII character set, to start with. First, though, Ian had to find a copy of the source, since Glenn had tossed it away as not worth the disk space. Ian and Glenn discussed what could be done with it, decided on a general re-write, and, 7 months later, ended up with 888 lines of code, 13 FIGfonts and documentation. This was FIGlet 2.0, the first real release.

To their great surprise, FIGlet took the net by storm. They received floods of "FIGlet is great!" messages and a new contributed FIGfont about once a week. To handle all the traffic, Ian quickly set up a mailing list, Daniel Simmons kindly offered space for an FTP site, several people volunteered to port FIGlet to non-Unix operating systems, ...and bug reports poured in.

Because of these, and the need to make FIGlet more "international", Ian and Glenn released a new version of FIGlet which could handle non-ASCII character sets and right-to-left printing. This was FIGlet 2.1, which, in a couple of weeks, became figlet 2.1.1. This weighed in at 1314 lines, and there were over 60 FIGfonts.

By late 1996, FIGlet had quite a following of fans subscribing to its mailing list. It had been ported to MS-DOS, Macintosh, Amiga, Apple II GS, Atari ST, Acorn and OS/2. FIGlet had been further updated, and there were nearly 200 FIGfonts.

John Cowan and Paul Burton are two FIGlet fans who decided to create new versions. While John wrote FIGlet version 2.2 using C, Paul wrote FIGWin 1.0, the first true GUI (Windows) implementation of FIGlet, using Visual Basic. John and Paul worked together to add new features to FIGfont files which could be read by both programs, and together wrote this document, which we hope helps to establish consistency in FIGfonts and help with the creation of future FIGdrivers. FIGlet 2.2 has about 4800 lines of code, of which over half is a support library for reading compressed files.

FIGlet 2.2 and FIGWin 1.0 both allow greater flexibility by use of new information which can be contained in FIGfont files without interfering with the function of older FIGdrivers.

> ⚠️ **HISTORICAL NOTE**: The Macintosh version of FIGlet (as of 1997) was still command-line driven, and a GUI version was in demand. The FIGlet C code was written to be easily plugged into a GUI shell.

## "Layout Modes" 📐

A FIGdriver may arrange FIGcharacters using one of three "layout modes", which define the spacing between FIGcharacters. The layout mode for the horizontal axis may differ from the layout mode for the vertical axis. A default choice is defined for each axis by every FIGfont.

```ascii
┌────────────────────────────────────────────────────────────┐
│ LAYOUT MODES VISUALIZED:                                   │
│                                                            │
│ ┌─────┐ ┌─────┐  │  ┌─────┐┌─────┐  │  ┌─────┐─────┐       │
│ │  A  │ │  B  │  │  │  A  ││  B  │  │  │  A  │  B  │       │
│ └─────┘ └─────┘  │  └─────┘└─────┘  │  └─────┘─────┘       │
│ Full Size        │  Fitting Only    │  Smushing            │
└────────────────────────────────────────────────────────────┘
```

The three layout modes are:

### Full Size (Separately called "Full Width" or "Full Height")

Represents each FIGcharacter occupying the full width or height of its arrangement of sub-characters as designed.

### Fitting Only (Separately called "Kerning" or "Vertical Fitting")

Moves FIGcharacters closer together until they touch. Typographers use the term "kerning" for this phenomenon when applied to the horizontal axis, but fitting also includes this as a vertical behavior, for which there is apparently no established typographical term.

### Smushing (Same term for both axes)

Moves FIGcharacters one step closer after they touch, so that they partially occupy the same space. A FIGdriver must decide what sub-character to display at each junction. There are two ways of making these decisions: by controlled smushing or by universal smushing.

- **Controlled smushing** uses a set of "smushing rules" selected by the designer of a FIGfont. (See "Smushing Rules" below.) Each rule is a comparison of the two sub-characters which must be joined to yield what to display at the junction. Controlled smushing will not always allow smushing to occur, because the compared sub-characters may not correspond to any active rule. Wherever smushing cannot occur, fitting occurs instead.

- **Universal smushing** simply overrides the sub-character from the earlier FIGcharacter with the sub-character from the later FIGcharacter. This produces an "overlapping" effect with some FIGfonts, wherein the latter FIGcharacter may appear to be "in front".

A FIGfont which does not specify any smushing rules for a particular axis indicates that universal smushing is to occur when smushing is requested. Therefore, it is not possible for a FIGfont designer to "forbid" smushing. However, there are ways to ensure that smushing does not cause a FIGfont to be illegible when smushed. This is especially important for smaller FIGfonts. (See "Hardblanks" for details.)

> 🔍 **TECHNICAL NOTE**: For vertical fitting or smushing, entire lines of output FIGcharacters are "moved" as a unit.

Not all FIGdrivers do vertical fitting or smushing. At present, FIGWin 1.0 does, but FIGlet 2.2 does not. Further, while FIGlet 2.2 allows the user to override the FIGfont designer's set of smushing rules, FIGWin 1.0 does not.

> ⚙️ **IMPLEMENTATION NOTE**: In FIGlet versions prior to 2.2, the term "smushmode" was used to mean the layout mode, and this term further included the smushing rules (if any) to be applied. However, since the layout mode may or may not involve smushing, more recent documentation uses more precise terminology.

# Advanced Typography Control

## "Smushing Rules" 🧩

> _"Precise character interaction creates typographic elegance."_ — Eidosian Typography Principle 2.3

Again, smushing rules are for controlled smushing. If none are defined to be active in a FIGfont, universal smushing occurs instead.

Generally, if a FIGfont is "drawn at the borders" using sub-characters "-\_|/\[]{}()<>", you will want to use controlled smushing by selecting from the rules below. Otherwise, if your FIGfont uses a lot of other sub-characters, do not select any rules and universal smushing will occur instead. (See "Hardblanks" below if your FIGfont is very small and would become illegible if smushed.) Experimentation is the best way to make these decisions.

```ascii
┌───────────────────────────────────────────────────────────────────────────┐
│ 🔍 RULE SELECTION GUIDANCE:                                              │
│                                                                           │
│ Border-drawn fonts (using "-_|/\[]{}()<>") → Use controlled smushing      │
│ Complex character fonts                    → Universal smushing preferred │
│ Small/compact fonts                        → Consider hardblanks          │
│                                                                           │
│ 🧪 Always experiment with different combinations to find optimal results! │
└───────────────────────────────────────────────────────────────────────────┘
```

There are six possible horizontal smushing rules and five possible vertical smushing rules. Below is a description of all of the rules.

> ⚠️ **NOTE**: Ignore the "code values" for now. They are explained later in the document.

### The Six Horizontal Smushing Rules

#### Rule 1: EQUAL CHARACTER SMUSHING (code value 1)

Two sub-characters are smushed into a single sub-character if they are the same. This rule does not smush hardblanks. (See "Hardblanks" below.)

#### Rule 2: UNDERSCORE SMUSHING (code value 2)

An underscore ("\_") will be replaced by any of: "|", "/", "\\", "[", "]", "{", "}", "(", ")", "<" or ">".

#### Rule 3: HIERARCHY SMUSHING (code value 4)

A hierarchy of six classes is used: "|", "/\\", "[]", "{}", "()", and "<>". When two smushing sub-characters are from different classes, the one from the latter class will be used.

#### Rule 4: OPPOSITE PAIR SMUSHING (code value 8)

Smushes opposing brackets ("[]" or "]["), braces ("{}" or "}{") and parentheses ("()" or ")(") together, replacing any such pair with a vertical bar ("|").

#### Rule 5: BIG X SMUSHING (code value 16)

Smushes "/\\" into "|", "\\/" into "Y", and "><" into "X". Note that "<>" is not smushed in any way by this rule. The name "BIG X" is historical; originally all three pairs were smushed into "X".

#### Rule 6: HARDBLANK SMUSHING (code value 32)

Smushes two hardblanks together, replacing them with a single hardblank. (See "Hardblanks" below.)

### The Five Vertical Smushing Rules

#### Rule 1: EQUAL CHARACTER SMUSHING (code value 256)

Same as horizontal smushing rule 1.

#### Rule 2: UNDERSCORE SMUSHING (code value 512)

Same as horizontal smushing rule 2.

#### Rule 3: HIERARCHY SMUSHING (code value 1024)

Same as horizontal smushing rule 3.

#### Rule 4: HORIZONTAL LINE SMUSHING (code value 2048)

Smushes stacked pairs of "-" and "\_", replacing them with a single "=" sub-character. It does not matter which is found above the other. Note that vertical smushing rule 1 will smush IDENTICAL pairs of horizontal lines, while this rule smushes horizontal lines consisting of DIFFERENT sub-characters.

#### Rule 5: VERTICAL LINE SUPERSMUSHING (code value 4096)

This one rule is different from all others, in that it "supersmushes" vertical lines consisting of several vertical bars ("|"). This creates the illusion that FIGcharacters have slid vertically against each other. Supersmushing continues until any sub-characters other than "|" would have to be smushed. Supersmushing can produce impressive results, but it is seldom possible, since other sub-characters would usually have to be considered for smushing as soon as any such stacked vertical lines are encountered.

```ascii
┌────────────────────────────────────────────────────────────────────┐
│ VERTICAL LINE SUPERSMUSHING VISUALIZATION:                         │
│                                                                    │
│ Original:                │ After Supersmushing:                    │
│  ┌─────┐                 │  ┌─────┐                                │
│  │  A  │                 │  │  A  │                                │
│  └─────┘                 │  └─────┘                                │
│    │                     │    │                                    │
│    │                     │  ┌─┴───┐                                │
│  ┌─┴───┐                 │  │  B  │                                │
│  │  B  │                 │  └─────┘                                │
│  └─────┘                 │                                         │
└────────────────────────────────────────────────────────────────────┘
```

## "Hardblanks" 🔲

A hardblank is a special sub-character which is displayed as a blank (space) in rendered FIGures, but is treated more like a "visible" sub-character when fitting or smushing horizontally. Therefore, hardblanks keep adjacent FIGcharacters a certain distance apart.

> 🔍 **TECHNICAL NOTE**: Hardblanks act the same as blanks for vertical operations.

### Hardblanks have three purposes

#### 1️⃣ Hardblanks are used to create the blank (space) FIGcharacter

Usually the space FIGcharacter is simply one or two vertical columns of hardblanks. Some slanted FIGfonts have a diagonal arrangement of hardblanks instead.

#### 2️⃣ Hardblanks can prevent "unreasonable" fitting or smushing

Normally when fitting or smushing, the blank (space) sub-character is considered "vacant space". In the following example, a capital "C" FIGcharacter is smushed with a "minus" FIGcharacter.

```
        ______                        ______
       / ____/                       / ____/
      / /      ____  >>-Becomes->   / /  ____
     / /___   /___/                / /__/___/
     \____/                        \____/
```

The FIGure above looks like a capital G. To prevent this, a FIGfont designer might place a hardblank in the center of the capital C. In the following example, the hardblank is represented as a "$":

```
        ______                        ______
       / ____/                       / ____/
      / /  $   ____  >>-Becomes->   / /   ____
     / /___   /___/                / /___/___/
     \____/                        \____/
```

Using hardblanks in this manner ensures that FIGcharacters with a lot of empty space will not be unreasonably "invaded" by adjacent FIGcharacters. Generally, FIGcharacters such as capital C, L or T, or small punctuation marks such as commas, may contain hardblanks, since they may contain a lot of vacant space which is "accessible" from either side.

#### 3️⃣ Hardblanks can prevent smushing from making FIGfonts illegible

This legitimate purpose of hardblanks is often overused. If a FIGfont designer is absolutely sure that smushing "visible" sub-characters would make their FIGfont illegible, hardblanks may be positioned at the end of each row of sub-characters, against the visible sub-characters, creating a barrier.

With older FIGdrivers, using hardblanks for this purpose meant that FIGcharacters would have to be separated by at least one blank in output FIGures, since only a hardblank could smush with another hardblank. However with the advent of universal smushing, this is no longer necessary. Hardblanks ARE overriden by any visible sub-character when performing universal smushing. Hardblanks still represent a "stopping point", but only AFTER their locations are occupied.

> ⚠️ **IMPORTANT**: Earlier it was stated that universal smushing overrides the sub-character from the former FIGcharacter with the sub-character from the latter FIGcharacter. Hardblanks (and blanks or spaces) are the exception to this rule; they will always be overriden by visible sub-characters, regardless of which FIGcharacter contains the hardblank. This ensures that no visible sub-characters "disappear".

Therefore, one can design a FIGfont with a default behavior of universal smushing, while the output FIGure would LOOK like the effect of fitting, or even full size if additional hardblanks are used. If a user "scales down" the layout mode to fitting, the result would look like "extra spacing" between FIGcharacters.

Taking this concept further, a FIGcharacter may also include extra blanks (spaces) on the left side of each FIGcharacter, which would define the FIGcharacter's width as slightly larger than required for the visible sub-characters and hardblanks. With such a FIGfont, a user who further "scales down" the layout mode to full size would see even greater spacing.

These techniques prevent horizontal smushing from causing a FIGfont to become illegible, while offering greater flexibility of output to users.

> 🔄 **DESIGN PHILOSOPHY NOTE**: These techniques cannot be used to prevent vertical smushing of visible sub-characters, since hardblanks are not respected in the vertical axis. Although it is possible to select only one vertical smushing rule which involves only sub-characters which are not used in your FIGfont, it is recommended that you do NOT do so. In our opinion, most users would prefer to get what they ask for, rather than being told, in effect: "I, the FIGfont designer, have decided that you wouldn't like the results of vertical smushing, so I have prevented you from trying it." Instead, we recommend setting the default behavior to either fitting or full height, and either allowing universal smushing, or selecting vertical smushing rules which seem most appropriate. A user of your FIGfont will quickly see why you did not choose smushing as the default vertical layout mode, and will agree with you.

```markdown
    ┌────────────────────────────────────────────────────────────────────────┐
    │ 💡 TYPOGRAPHY TIP:                                                     │
    │                                                                        │
    │ When designing FIGfonts, think of hardblanks as the "personal space"   │
    │ of your characters—they establish boundaries that respect the integrity │
    │ of each glyph while still allowing for dynamic layout options.         │
    └────────────────────────────────────────────────────────────────────────┘
```

# Character Sets and Character Encoding

## "Character Sets" and "Character Codes" 🔣

> _"Digital typography begins with understanding the numerical foundation of characters."_ — Eidosian Typography Principle 3.6

When you type using your keyboard, you are actually sending your computer a series of numbers. Each number must be interpreted by your computer so that it knows what character to display. The computer uses a list of definitions, called a "character set". The numbers which represent each character are called "character codes".

```ascii
┌───────────────────────────────────────────────────────────────┐
│ KEY PRESS → CHARACTER CODE → CHARACTER SET → DISPLAYED GLYPH  │
└───────────────────────────────────────────────────────────────┘
```

There are many character sets, most of which are internationally accepted as standards. By far, the most common character set is ASCII, which stands for "American Standard Code for Information Interchange". ASCII identifies its characters with codes ranging from 0 to 127.

> 🔍 **HISTORICAL NOTE**: The term "ASCII art" has become well-understood to mean artistic images which consist of characters on your screen (such as FIGures).

For a list of the printable ASCII characters with the corresponding codes, see the section "REQUIRED CHARACTERS" below. The other ASCII codes in the range of 0 through 31 are "control characters" such as carriage-return (code 13), linefeed/newline (code 10), tab (code 9), backspace (code 8) or null (code 0). Code 127 is a delete in ASCII.

### Character Set Fundamentals 📊

Getting more technical for just a moment: A byte consisting of 8 bits (eight 1's or 0's) may represent a number from 0 to 255. Therefore, most computers have DIRECT access to 256 characters at any given time. A character set which includes 256 characters is called an 8-bit character set.

```ascii
┌──────────────────────────────────────────────────────────┐
│ CHARACTER SET CAPACITY:                                  │
│                                                          │
│ ASCII       → 7-bit → 128 characters (0-127)             │
│ 8-bit sets  → 8-bit → 256 characters (0-255)             │
│ Unicode     → Multi-byte → Over 143,000 characters       │
└──────────────────────────────────────────────────────────┘
```

For Latin-based languages, ASCII is almost always the first half of a larger 8-bit character set. Latin-1 is the most common example of an 8-bit character set. Latin-1 includes all of ASCII, and adds characters with codes from 128 to 255 which include umlauted ("double-dotted") letters and characters with various other accents. In the United States, Windows and most Unix systems have Latin-1 directly available.

Most modern systems allow the possibility of changing 8-bit character sets. On Windows systems, character sets are referred to as "code pages". There are many other character sets which are not mentioned here. DOS has its own character set (which also has international variants) that includes graphics characters for drawing lines. It is also an extension of ASCII.

For some languages, 8-bit character sets are insufficient, particularly on East Asian systems. Therefore, some systems allow 2 bytes for each character, which multiplies the 256 possibilities by 256, resulting in 65536 possible characters. (Much more than the world will ever need.)

### Unicode: The Universal Character Set 🌐

Unicode is a character set standard which is intended to fulfill the worldwide need for a single character set which includes all characters used worldwide. Unicode includes character codes from 0 to 65535, although at present, only about 22,000 characters have been officially assigned and named by the Unicode Consortium. The alphabets and other writing systems representable with Unicode include all Latin-alphabet systems, Greek, Russian and other Cyrillic-alphabet systems, Hebrew, Arabic, the various languages of India, Chinese, Japanese, Korean, and others. The existing Unicode symbols include chess pieces, astrological signs, gaming symbols, telephones, pointing fingers, etc. — just about any type of FIGcharacter you may wish to create. Unicode is constantly (but slowly) being extended to handle new writing systems and symbols. Information on Unicode is available at <http://www.unicode.org> and at ftp://unicode.org.

Unicode, Latin-1, and ASCII all specify the same meanings for overlapping character codes: ASCII 65 = Latin-1 65 = Unicode 65 = "A", formally known as "LATIN CAPITAL LETTER A".

```ascii
┌────────────────────────────────────────────────────────────────┐
│ CHARACTER CODE COMPATIBILITY:                                  │
│                                                                │
│       ┌────────┐                                               │
│       │ ASCII  │ 0-127                                         │
│       └────────┘                                               │
│          ⊂                                                     │
│     ┌──────────┐                                               │
│     │ Latin-1  │ 0-255                                         │
│     └──────────┘                                               │
│          ⊂                                                     │
│  ┌──────────────────────────┐                                  │
│  │       Unicode            │ 0-143,859 (and growing)          │
│  └──────────────────────────┘                                  │
└────────────────────────────────────────────────────────────────┘
```

### Keyboard Maps and Character Access 🖮

Since a keyboard usually has only about 100 keys, your computer may contain a program called a "keyboard map", which will interpret certain keystrokes or combinations of keystrokes as different character codes. Keyboard maps use "mapping tables" to make these determinations. The appropriate keyboard activity for a given character code may involve several keystrokes. Almost all systems are capable of handling at least 8-bit character sets (containing 256 characters), so there is always an active keyboard map, at least for those characters which are not actually painted on the keys. (United States users may not even know that their computer can interpret special keystrokes. Such keystrokes may be something similar to holding down the ALT key while typing a character code on the numeric keypad. Try it!)

Below are characters 160 through 255, AS REPRESENTED ON YOUR SYSTEM.

```
       ������������������������������������������������
       ������������������������������������������������
```

> ⚠️ **IMPORTANT NOTE**: Depending on which character set is active on your system, you may see different characters. This document (like all computer documents) does not contains characters per se, only bytes. What you see above is your particular computer's representation of these byte values. In other words, your active character set. However, if it is Latin-1, the first visible character is an inverted "!", and the last is an umlauted "y". Although we can safely assume your computer has ASCII, it does not necessarily have the Latin-1 character set active.

## Character Sets and FIGfonts 🎭

What does all this have to do with FIGfonts?

First, it should be evident that it is best to use only ASCII characters for sub-characters when possible. This will ensure portability to different platforms.

```ascii
┌───────────────────────────────────────────────────────────────┐
│ 💡 PORTABILITY TIP:                                          │
│                                                               │
│ Stick to ASCII (0-127) for maximum compatibility across       │
│ all systems and character sets when designing FIGfonts.       │
└───────────────────────────────────────────────────────────────┘
```

FIGlet has gained international popularity, but early versions were made to handle only FIGcharacters with assigned character codes corresponding to ASCII. So, over the years there have been four methods used to create "virtual mapping tables" within the program itself:

1. The first method was simply to create FIGcharacters which do not look like the ASCII character set implies. For example, a FIGfont might contain Greek letters, and within its comments, it may say, "If you type A, you'll get a Greek Alpha" etc. With the advent of newer features, it is preferable not to use this method. Instead, when possible, add new FIGcharacters to existing FIGfonts or create new FIGfonts with FIGcharacters coded to match the expectations of ASCII/Latin-1/Unicode, and create an appropriate control file. (See "CONTROL FILES" below.) Remember that Unicode includes almost any character for which you may want to create a FIGcharacter.

2. The second method was very specific, to accommodate the German audience. A special option was added to the FIGlet program which would re-route input characters "[", "\", and "]" to umlauted A, O and U, while "{", "|", and "}" would become the respective lowercase versions of these. Also, "~" was made to become the s-z character when this special option was used. This was called "the -D option." The addition of this feature meant that all compatible FIGfonts must contain these Deutsch (German) FIGcharacters, in addition to the ASCII FIGcharacters. Although this option is still available in the most recent version, it is no longer necessary, as the same result can be achieved by the newer features described below. However, the requirement for Deutsch FIGcharacters remains for backward compatibility. (Or at least zero-width FIGcharacters in their place.)

3. Later, FIGlet was made to accept control files, which are quite literally a form of mapping table. (See "CONTROL FILES" below.) This was a significant advance for internationalization.

4. FIGlet 2.2 can now accept specially encoded formats of input text which imply more than one byte per character.

## Creating FIGfonts 🎨

> 🧰 **TOOL NOTE**: FIGWin 1.0 is packaged with a program called FIGfont Editor for Windows 1.0, which is just that. There is no need to read further if you intend to use it. However, the section "CONTROL FILES" below is still relevant.

Since a FIGfont file is a text file, it can be created with any text editing program on any platform, and will still be compatible with FIGdrivers on all operating systems, except that the bytes used to indicate the end of each text line may vary. (PC's use carriage return and linefeed at the end of each line, Macintosh uses carriage return only, and UNIX uses linefeed only.)

This minor difference among operating systems is handled easily by setting your FTP program to ASCII mode during upload or download. So there is no need to be concerned about this as long as you remember to do this during file transfer.

The filename of a FIGfont file must end with ".flf", which stands for "<F>IG<L>ettering <F>ont". The first part of the filename should contain only letters, and should be lowercase on operating systems which permit case sensitive filenames. The filename should be unique in the first 8 characters, since some older file systems truncate longer filenames.

It is easier to modify an existing FIGfont than it is to create a new one from scratch. The first step is to read and understand this document. You may want to load "standard.flf" or another FIGfont into a text editor as an example while you read.

A FIGfont file contains three portions: a header line, comments, and FIGcharacter data.

### The Header Line 📋

The header line gives information about the FIGfont. Here is an example showing the names of all parameters:

```
flf2a$ 6 5 20 15 3 0 143 229    NOTE: The first five characters in
  |  | | | |  |  | |  |   |     the entire file must be "flf2a".
 /  /  | | |  |  | |  |   \

Signature / / | | | | | \ Codetag_Count
Hardblank / / | | | \ Full_Layout*
Height / | | \ Print_Direction
Baseline / \ Comment_Lines
Max_Length Old_Layout*
```

- The two layout parameters are closely related and fairly complex. (See "INTERPRETATION OF LAYOUT PARAMETERS".)

For those desiring a quick explanation, the above line indicates that this FIGfont uses "$" to represent the hardblank in FIGcharacter data, it has FIGcharacters which are 6 lines tall, 5 of which are above the baseline, no line in the FIGfont data is more than 20 columns wide, the default horizontal layout is represented by the number 15, there are 3 comment lines, the default print direction for this FIGfont is left-to-right, a complete description of default and possible horizontal and vertical layouts is represented by the number 143, and there are 229 code-tagged characters.

The first seven parameters are required. The last three (Direction, Full_Layout, and Codetag_Count) are not. This allows for backward compatibility with older FIGfonts, but a FIGfont without these parameters would force a FIGdriver to "guess" (by means not described in this document) the information it would expect to find in Full_Layout. For this reason, inclusion of all parameters is strongly recommended.

Future versions of this standard may add more parameters after Codetag_Count.

### Parameter Descriptions 🔍

#### Signature

The signature is the first five characters: "flf2a". The first four characters "flf2" identify the file as compatible with FIGlet version 2.0 or later (and FIGWin 1.0). The "a" is currently ignored, but cannot be omitted. Different characters in the "a" location may mean something in future versions of this standard. If so, you can be sure your FIGfonts will still work if this character is "a".

#### Hardblank

Immediately following the signature is the hardblank character. The hardblank character in the header line defines which sub-character will be used to represent hardblanks in the FIGcharacter data.

By convention, the usual hardblank is a "$", but it can be any character except a blank (space), a carriage-return, a newline (linefeed) or a null character. If you want the entire printable ASCII set available to use, make the hardblank a "delete" character (character code 127). With the exception of delete, it is inadvisable to use non-printable characters as a hardblank.

#### Height

The Height parameter specifies the consistent height of every FIGcharacter, measured in sub-characters. Note that ALL FIGcharacters in a given FIGfont have the same height, since this includes any empty space above or below. This is a measurement from the top of the tallest FIGcharacter to the bottom of the lowest hanging FIGcharacter, such as a lowercase g.

#### Baseline

The Baseline parameter is the number of lines of sub-characters from the baseline of a FIGcharacter to the top of the tallest FIGcharacter. The baseline of a FIGfont is an imaginary line on top of which capital letters would rest, while the tails of lowercase g, j, p, q, and y may hang below. In other words, Baseline is the height of a FIGcharacter, ignoring any descenders.

This parameter does not affect the output of FIGlet 2.2 or FIGWin 1.0, but future versions or other future FIGdrivers may use it. The Baseline parameter should be correctly set to reflect the true baseline as described above. It is an error for Baseline to be less than 1 or greater than the Height parameter.

#### Max_Length

The Max_Length parameter is the maximum length of any line describing a FIGcharacter. This is usually the width of the widest FIGcharacter, plus 2 (to accommodate endmarks as described later). However, you can (and probably should) set Max_Length slightly larger than this as a safety measure in case your FIGfont is edited to include wider FIGcharacters. FIGlet (but not FIGWin 1.0) uses this number to minimize the memory taken up by a FIGfont, which is important in the case of FIGfonts with many FIGcharacters.

#### Old_Layout

(See "Interpretation of Layout Parameters" below.)

#### Comment_Lines

Between the first line and the actual FIGcharacters of the FIGfont are the comment lines. The Comment_Lines parameter specifies how many lines there are. Comments are optional, but recommended to properly document the origin of a FIGfont.

#### Print_Direction

The Print_Direction parameter tells which direction the font is to be printed by default. A value of 0 means left-to-right, and 1 means right-to-left. If this parameter is absent, 0 (left-to-right) is assumed. Print_Direction may not specify vertical print, although FIGdrivers are capable of vertical print. Versions of FIGlet prior to 2.1 ignore this parameter.

#### Full_Layout

(See "Interpretation of Layout Parameters" just below.)

#### Codetag_Count

Indicates the number of code-tagged (non-required) FIGcharacters in this FIGfont. This is always equal to the total number of FIGcharacters in the font minus 102. This parameter is typically ignored by FIGdrivers, but can be used to verify that no characters are missing from the end of the FIGfont. The chkfont program will display the number of codetagged characters in the FIGfont on which it is run, making it easy to insert this parameter after a FIGfont is written.

## Interpretation of Layout Parameters 🧮

Full_Layout describes ALL information about horizontal and vertical layout: the default layout modes and potential smushing rules, even when smushing is not a default layout mode.

Old_Layout does not include all of the information desired by the most recent FIGdrivers, which is the inspiration for the creation of the new Full_Layout parameter. Old_Layout is still required for backward compatibility, and FIGdrivers must be able to interpret FIGfonts which do not have the Full_Layout parameter.

Versions of FIGlet prior to 2.2 do not recognize the Full_Layout parameter. Documentation accompanying FIGlet versions prior to 2.2 refer to Old_Layout as "smushmode", which is somewhat misleading since it can indicate layout modes other than smushing.

Old_Layout and Full_Layout must contain some redundant information.

### Layout Parameters Code Values 🔢

Setting the layout parameters is a matter of adding numbers together ("code values"). Here's a chart of the meanings of all code values:

#### Full_Layout (Legal values 0 to 32767)

```ascii
┌───────────────────────────────────────────────────────────┐
│ HORIZONTAL PARAMETERS:                                    │
│    1 - Apply horizontal smushing rule 1 when smushing     │
│    2 - Apply horizontal smushing rule 2 when smushing     │
│    4 - Apply horizontal smushing rule 3 when smushing     │
│    8 - Apply horizontal smushing rule 4 when smushing     │
│   16 - Apply horizontal smushing rule 5 when smushing     │
│   32 - Apply horizontal smushing rule 6 when smushing     │
│   64 - Horizontal fitting (kerning) by default            │
│  128 - Horizontal smushing by default (Overrides 64)      │
│                                                           │
│ VERTICAL PARAMETERS:                                      │
│  256 - Apply vertical smushing rule 1 when smushing       │
│  512 - Apply vertical smushing rule 2 when smushing       │
│ 1024 - Apply vertical smushing rule 3 when smushing       │
│ 2048 - Apply vertical smushing rule 4 when smushing       │
│ 4096 - Apply vertical smushing rule 5 when smushing       │
│ 8192 - Vertical fitting by default                        │
│ 16384 - Vertical smushing by default (Overrides 8192)     │
└───────────────────────────────────────────────────────────┘
```

When no smushing rules are included in Full_Layout for a given axis, the meaning is that universal smushing shall occur, either by default or when requested.

#### Old_Layout (Legal values -1 to 63)

```ascii
┌──────────────────────────────────────────────────┐
│  -1 - Full-width layout by default               │
│   0 - Horizontal fitting (kerning) by default*   │
│   1 - Apply horizontal smushing rule 1 by default│
│   2 - Apply horizontal smushing rule 2 by default│
│   4 - Apply horizontal smushing rule 3 by default│
│   8 - Apply horizontal smushing rule 4 by default│
│  16 - Apply horizontal smushing rule 5 by default│
│  32 - Apply horizontal smushing rule 6 by default│
└──────────────────────────────────────────────────┘
```

> ⚠️ **IMPORTANT NOTE**: When Full_Layout indicates UNIVERSAL smushing as a horizontal default (i.e., when none of the code values of horizontal smushing rules are included and code value 128 is included in Full_Layout), Old_Layout must be set to 0 (zero). Older FIGdrivers which cannot read the Full_Layout parameter are also incapable of universal smushing. Therefore they would be directed to the "next best thing", which is horizontal fitting (kerning).

> 🚫 **WARNING**: You should NOT add the -1 value to any positive code value for Old_Layout. This would be a logical contradiction.

### Layout Parameter Consistency Rules 📏

The following rules establish consistency between Old_Layout and Full_Layout:

#### If full width is to be the horizontal default

- Old_Layout must be -1.
- Full_Layout must NOT include code values 64 nor 128.

#### If horizontal fitting (kerning) is to be default

- Old_Layout must be 0.
- Full_Layout must include code value 64.
- Full_Layout must NOT include code value 128.

#### If CONTROLLED smushing is to be the horizontal default

- Old_Layout must be a positive number, represented by the added code values of all desired horizontal smushing rules.
- Full_Layout must include the code values for the SAME set of horizontal smushing rules as included in Old_Layout.
- Full_Layout must include code value 128.

#### If UNIVERSAL smushing is to be the horizontal default

- Old_Layout must be 0.
- Full_Layout must include code value 128.
- Full_Layout must NOT include any code value under 64.

In general terms, if Old_Layout specifies horizontal smushing rules, Full_Layout must specify the same set of horizontal rules, and both must indicate the same horizontal default layout mode.

## Setting Layout Parameters Step-by-Step 📝

The following process will yield correct and consistent values for the two layout parameters:

### Step 1: Start with 0 for both numbers

- Write "Old_Layout" and "Full_Layout" on a piece of paper.
- Write the number 0 next to each.
- The number 0 may be crossed out and changed several times below.

### Step 2: Set the DEFAULT HORIZONTAL LAYOUT MODE

```ascii
┌──────────────────────────────────────────────────────────┐
│ If you want to use FULL WIDTH as the default:            │
│   - Make Old_Layout -1                                   │
│                                                          │
│ If you want to use HORIZONTAL FITTING (kerning) as       │
│ the default:                                             │
│   - Make Full_Layout 64                                  │
│                                                          │
│ If you want to use HORIZONTAL SMUSHING as the default:   │
│   - Make Full_Layout 128                                 │
└──────────────────────────────────────────────────────────┘
```

### Step 3: Specify HOW TO SMUSH HORIZONTALLY WHEN SMUSHING

```ascii
┌──────────────────────────────────────────────────────────┐
│ If you want to use UNIVERSAL smushing for horizontal:    │
│   - Skip to step 4                                       │
│                                                          │
│ If you want to use CONTROLLED smushing for horizontal:   │
│   - Add up code values for desired horizontal rules:     │
│     EQUAL CHARACTER SMUSHING          1                  │
│     UNDERSCORE SMUSHING               2                  │
│     HIERARCHY SMUSHING                4                  │
│     OPPOSITE PAIR SMUSHING            8                  │
│     BIG X SMUSHING                   16                  │
│     HARDBLANK SMUSHING               32                  │
│                                                          │
│ If Full_Layout is currently 128:                         │
│   - Change Old_Layout to the horizontal smushing total   │
│   - Increase Full_Layout by this total                   │
│                                                          │
│ If Full_Layout is currently 0 or 64:                     │
│   - Increase Full_Layout by the horizontal smushing total│
└──────────────────────────────────────────────────────────┘
```

### Step 4: Set the DEFAULT VERTICAL LAYOUT MODE

```ascii
┌──────────────────────────────────────────────────────────┐
│ If you want to use FULL HEIGHT as the default:           │
│   - Skip to step 5                                       │
│                                                          │
│ If you want to use VERTICAL FITTING as the default:      │
│   - Increase Full_Layout by 8192                         │
│                                                          │
│ If you want to use VERTICAL SMUSHING as the default:     │
│   - Increase Full_Layout by 16384                        │
└──────────────────────────────────────────────────────────┘
```

### Step 5: Specify HOW TO SMUSH VERTICALLY WHEN SMUSHING

```ascii
┌──────────────────────────────────────────────────────────┐
│ If you want to use UNIVERSAL smushing for vertical:      │
│   - Skip to step 6                                       │
│                                                          │
│ If you want to use CONTROLLED smushing for vertical:     │
│   - Add up code values for desired vertical rules:       │
│     EQUAL CHARACTER SMUSHING        256                  │
│     UNDERSCORE SMUSHING             512                  │
│     HIERARCHY SMUSHING             1024                  │
│     HORIZONTAL LINE SMUSHING       2048                  │
│     VERTICAL LINE SUPERSMUSHING    4096                  │
│                                                          │
│   - Increase Full_Layout by the vertical smushing total  │
└──────────────────────────────────────────────────────────┘
```

### Step 6: You're done

The resulting value of Old_Layout will be a number from -1 to 63.
The resulting value of Full_Layout will be a number from 0 and 32767.

```ascii
┌─────────────────────────────────────────────────────────────┐
│ 💡 LAYOUT PARAMETER INSIGHT:                               │
│                                                             │
│ Think of Old_Layout as the legacy setting that handles      │
│ horizontal aspects only, while Full_Layout is the complete  │
│ control system for both horizontal and vertical layouts.    │
│                                                             │
│ When setting parameters, always ensure they're consistent   │
│ to maintain backward compatibility while enabling advanced  │
│ layout features in modern FIGdrivers.                       │
└─────────────────────────────────────────────────────────────┘
```

# FIGfont Comments and Character Data

## FIGfont Comments 📝

After the header line come the FIGfont comments. These comments can span as many lines as you wish, but should at least include your name and email address. Here's an example that also shows the header line:

```
flf2a$ 6 5 20 15 3 0 143
Example by Glenn Chappell <ggc@uiuc.edu> 8/94
Permission is hereby given to modify this font, as long as the
modifier's name is placed on a comment line.
```

While comments aren't required, they're appreciated and help document the origin and authorship of your font. Remember to adjust the `Comment_Lines` parameter in the header line accordingly as you add lines to your comments. Important note: blank lines DO count toward the total comment line count!

## FIGcharacter Data Structure 🧩

The FIGcharacter data begins immediately after the comments and continues to the end of the file. The data consists of ASCII characters arranged to form larger visual characters when displayed.

### Basic Data Structure ⚙️

The sub-characters in the file are given exactly as they should be output, with two important exceptions:

1. **Hardblanks** should be represented using the hardblank character specified in the header line, not regular spaces.
2. **Endmark characters** appear at the end of each line, defining the width of each FIGcharacter.

In most FIGfonts, the endmark character is either `@` or `#`. The FIGdriver automatically eliminates the last block of consecutive identical characters from each line when reading the font. By convention, the last line of a FIGcharacter has two endmarks, while all the rest have one. This makes it easy to visually identify where FIGcharacters begin and end. No line should have more than two endmarks.

```ascii
┌────────────────────────────────────────────────┐
│ FIGCHARACTER DATA STRUCTURE:                   │
│                                                │
│ • Every character has same height (e.g., 6)    │
│ • First 102 characters are required            │
│ • Additional characters need code tags         │
│ • Last line of each character has double       │
│   endmarks                                     │
│ • All lines for one character must have        │
│   consistent width                             │
└────────────────────────────────────────────────┘
```

### Example FIGcharacters 🔠

Here's an example of the first few FIGcharacters from `small.flf`. The vertical line `|` represents the left margin of your editor (not part of the FIGfont). Hardblanks are represented as `$` in this example:

```
                        |$@
                        |$@
           blank/space  |$@
                        |$@
                        |$@@
                        | _ @
                        || |@
     exclamation point  ||_|@
                        |(_)@
                        |   @@
                        | _ _ @
                        |( | )@
          double quote  | V V @
                        |  $  @
                        |     @@
                        |   _ _   @
                        | _| | |_ @
           number sign  ||_  .  _|@
                        ||_     _|@
                        |  |_|_|  @@
                        |    @
                        | ||_@
           dollar sign  |(_-<@
                        |/ _/@
                        | || @@
```

Important points to note:

- Each FIGcharacter occupies the same number of lines (6 lines in this case), matching the Height parameter in the header.
- Every line within a FIGcharacter must have consistent width once the endmarks are removed.
- Pay attention to vertical alignment so characters line up properly when printed.
- If one of your last sub-characters is `@`, use a different endmark character (like `#`) for that FIGcharacter to avoid confusion.

## Required FIGcharacters 📋

Some FIGcharacters are required and must be presented in a specific order:

1. All printable ASCII characters (codes 32-126), including:

   - Space (32)
   - Letters A-Z (65-90) and a-z (97-122)
   - Numbers 0-9 (48-57)
   - Punctuation and special characters

2. Seven additional German characters in this order:

   - Umlauted "A", "O", "U", "a", "o", "u" (codes 196, 214, 220, 228, 246, 252)
   - Ess-zed (code 223) - looks like:

     ```
     ___
     / _ \
     | |/ /
     | |\ \
     | ||_/
     |_|

     ```

If you don't want to define all required FIGcharacters, you can create "empty" ones by placing endmarks flush with the left margin. The German characters are commonly created as empty. If your font includes only capital letters, please copy them to the appropriate lowercase positions rather than leaving them empty.

## Code-Tagged FIGcharacters 🏷️

After the required characters, you may add additional characters with any code from -2147483648 to +2147483647 (except -1, which is reserved). These must include a "code tag" line before each character that specifies the character code.

A code tag contains:

- The character code (decimal, octal, or hexadecimal)
- Whitespace
- An optional comment (usually the name of the character)

For example:

```
161  INVERTED EXCLAMATION MARK
 _ @
(_)@
| |@
| |@
|_|@
   @@
```

Number formats:

- Decimal: `161` (standard numbers)
- Octal: Preceded by `0` (e.g., `0241`)
- Hexadecimal: Preceded by `0x` or `0X` (e.g., `0xA1`)

For Unicode fonts, it's common to express codes in hexadecimal as that's the standard for Unicode specifications. For smaller character sets (below 256), decimal is commonly used.

### Special Cases for Character Codes 🚩

1. **Code 0**: Used as the "missing character" that displays when an undefined character is requested
2. **Codes 1-31**: Avoid using (ASCII control characters)
3. **Code 127**: Avoid (ASCII DELETE)
4. **Codes 128-159**: Avoid (Latin-1 control characters)
5. **Code -1**: Illegal for technical reasons

FIGfont designers might want to use negative codes (except -1) for special purposes like translation tables, especially in Unicode fonts.

```ascii
┌───────────────────────────────────────────────────────────┐
│ 💡 CHARACTER CODE BEST PRACTICES:                        │
│                                                           │
│ • Follow ASCII/Latin-1/Unicode conventions when possible  │
│ • Use code 0 for "missing character" fallback             │
│ • Express codes <256 in decimal                           │
│ • Express Unicode codes in hexadecimal                    │
│ • Avoid control character codes (1-31, 127, 128-159)      │
│ • Use negative codes for special purposes                 │
│ • Never use code -1 (reserved)                            │
└───────────────────────────────────────────────────────────┘
```

With these guidelines, you can create rich FIGfonts with characters from basic ASCII to complex Unicode glyphs, ensuring compatibility with various FIGdrivers and providing a great experience for users of your font.

# FIGfont Error Prevention and Control Files

## Avoiding Errors in FIGfonts 🚨

Creating error-free FIGfonts requires attention to several key details:

### Critical Requirements 📏

- **Consistent Height**: Every character in a font must have exactly the same height, matching the `Height` parameter in the header
- **Consistent Width**: All lines of a single FIGcharacter must have the same length once endmarks are removed
- **No Trailing Spaces**: Be extremely careful not to leave trailing spaces in your font file as FIGdrivers will interpret these as endmarks
- **Proper Space Character**: The blank (space) FIGcharacter should typically consist of one or two columns of hardblanks only, unless you're making a slanted font
  - Without hardblanks, space characters will disappear during kerning or smushing!

### Font Validation 🧪

- Use the `chkfont` program (part of the standard FIGlet package) to detect errors in your font
- FIGWin is less forgiving than FIGlet and will immediately report errors during font loading
- FIGlet might produce nonsensical output when given incorrectly formatted fonts, making errors harder to detect

```ascii
┌────────────────────────────────────────────────────┐
│ COMMON FIGFONT ERRORS:                            │
│                                                   │
│ • Inconsistent character height                   │
│ • Varying line lengths within characters          │
│ • Trailing spaces getting interpreted as endmarks │
│ • Missing hardblanks in space character           │
│ • Non-ASCII characters displaying inconsistently  │
└────────────────────────────────────────────────────┘
```

> ⚠️ **Important**: Remember that sub-characters outside the ASCII range may not display consistently across different systems!

## Control Files (.flc) 🔄

Control files let you map input characters to different FIGcharacter codes, enabling advanced character handling features.

### Purpose and Features 🎯

- Map input characters to different FIGfont character codes
- Enable access to characters beyond Latin-1 (codes > 255)
- Support multiple character encodings (DBCS, UTF-8, etc.)
- Create multiple transformation stages
- Control how FIGdrivers interpret input bytes

### File Format 📄

Control files use the `.flc` extension and contain command lines with several formats:

```
t inchar outchar                    # Transform single character
t inchar1-inchar2 outchar1-outchar2 # Transform character ranges
number number                       # Alternative transform syntax
f                                   # Separate transformation stages
h, j, b, u                          # Special input interpretation modes
g{0|1|2|3} {94|96|94x94} [char]     # ISO 2022 character set commands
g{L|R} {0|1|2|3}                    # ISO 2022 half assignment
```

Lines beginning with `#` and blank lines are treated as comments.

### Character Representation 📝

Characters can be specified in multiple ways:

- As literal characters: `A` represents code 65
- As numeric codes with backslash: `\65` also represents 65
- Using hexadecimal: `\0x100` represents code 256
- Using special escape sequences:
  - `\a` = code 7 (bell/alert)
  - `\b` = code 8 (backspace)
  - `\e` = code 27 (ESC)
  - `\f` = code 12 (form feed)
  - `\n` = code 10 (newline)
  - `\r` = code 13 (carriage return)
  - `\t` = code 9 (tab)
  - `\v` = code 11 (vertical tab)
  - `\\` = code 92 (backslash)
  - `\` = code 32 (space)

### Command Examples 📚

#### Basic Transformation

```
t # $       # Transforms # into $
t A-Z a-z   # Transforms A-Z into a-z (lowercase conversion)
```

#### Character Swapping

```
t A B       # A becomes B
t B A       # B becomes A (effectively swapping A and B)
```

#### Multi-Stage Transformations

```
t a-z A-Z   # First stage: lowercase to uppercase
f           # Separates transformation stages
t Q ~       # Second stage: Q (including converted q) becomes ~
```

### Extended Input Interpretation Commands 🌐

For FIGlet 2.2 and later, these commands control how input bytes are interpreted:

- `h` - HZ mode for Chinese text
- `j` - Shift-JIS mode (MS-Kanji)
- `b` - DBCS mode for Chinese/Korean text
- `u` - UTF-8 mode for Unicode characters

Each handles multi-byte characters differently, calculating character codes through specific formulas.

### ISO 2022 Support 🌍

Control files can configure ISO 2022 character set handling through `g` commands:

```
g 0 94 B    # Set G0 to ASCII (94-character set "B")
g 1 96 A    # Set G1 to Latin-1 top half (96-character set "A")
g R 2       # Interpret right-half bytes as G2
```

ISO 2022 allows complex character set switching through escape sequences, supporting various East Asian text encodings.

```ascii
┌──────────────────────────────────────────────────────────┐
│ PRACTICAL CONTROL FILE APPLICATIONS:                    │
│                                                         │
│ • Unicode access: Use UTF-8 mode for extended chars     │
│ • Case conversion: Transform uppercase to lowercase     │
│ • Character substitution: Replace unavailable chars     │
│ • Multi-language support: Switch between char sets      │
│ • Special effects: Remap characters for artistic output │
└──────────────────────────────────────────────────────────┘
```

The standard FIGlet distribution includes mapping tables for Latin-2 (ISO 8859-2), Latin-3 (ISO 8859-3), Latin-4 (ISO 8859-4), and Latin-5 (ISO 8859-9), designed to work with the standard.flf font.

> 💡 **Tip**: Control files are powerful tools for extending FIGlet's capabilities beyond basic ASCII art, enabling international character support and creative character remapping!

# FIGfont Standards and Capabilities for FIGdrivers

## Core Requirements for FIGdrivers 📋

### Licensing and Distribution 💸

FIGdrivers follow strict requirements regarding their availability and licensing:

- **Free Access** — All FIGdrivers must be freely available to the public via the internet
- **No Advertising** — Ads for other works must be limited to documentation only (max one screenful)
- **No Time Limitations** — FIGdrivers cannot disable themselves after a trial period
- **No Donation Requirements** — Donation requests are prohibited
- **No Pay-to-Upgrade** — Paid versions with enhanced FIGure creation capabilities are not allowed

```ascii
┌────────────────────────────────────────────────────────┐
│ FIGDRIVER COMMERCIAL PROHIBITIONS:                     │
│                                                        │
│ • No feature paywalls                                 │
│ • No time-limited versions                            │
│ • No mandatory donations                              │
│ • No in-program advertising                           │
│ • No paid version with enhanced capabilities          │
└────────────────────────────────────────────────────────┘
```

### Required Features and Naming 🏷️

- **Minimum Functionality** — Must process FIGfont files as described in this document
- **Optional Features** — Not required to implement control files, smushing, fitting, kerning, vertical operations, or multi-line output
- **Naming Convention** — Must include capitalized "FIG" in the name and have an incremental version number specific to its platform

### Backwards Compatibility Guidelines ♻️

When creating new versions of existing FIGdrivers:

- **Full Capability Preservation** — Must maintain all functionality from previous versions
- **Historical Documentation** — Must preserve comments about past version updates
- **Source Code Access** — Should provide source code to the public or at least to potential developers
- **Distinct Naming** — New programs for existing platforms must use different names

## FIGfont Format Evolution 📝

Changes to the FIGfont format require meeting all these conditions:

1. **ASCII Text Editability** — New format must be easily editable as ASCII text
2. **Sequential Versioning** — Must begin with "flf" followed by a sequential number
3. **Information Preservation** — Must retain all information from previous formats, including designer credits
4. **Cross-Platform Support** — Requires developers willing to port to key platforms (UNIX, DOS, Windows, Amiga)
5. **Command-line Access** — A C, Java, or other version must exist for command-line or internet-based use
6. **Backward Format Support** — Must read both old and new formats
7. **Migration Path** — All existing FIGfonts from the official distribution must be converted and available

```ascii
┌────────────────────────────────────────────────────────┐
│ FIGFONT FORMAT EVOLUTION REQUIREMENTS:                 │
│                                                        │
│ ✓ ASCII text format                                   │
│ ✓ Sequential versioning (flf#)                        │
│ ✓ Complete information retention                      │
│ ✓ Multi-platform implementation                       │
│ ✓ Command-line accessibility                          │
│ ✓ Backward compatibility                              │
│ ✓ Complete conversion of existing fonts               │
└────────────────────────────────────────────────────────┘
```

## Word Wrapping Behavior 📏

FIGdrivers handling multi-line output should implement these word wrapping behaviors:

- **Width Control** — Monitor blanks in input text to ensure FIGures fit within maximum allowed width
- **Manual Line Breaks** — Support user-specified linebreaks in addition to automatic wrapping
- **Platform-Specific Handling** — Properly handle different linebreak representations on various platforms
- **Blank Management** — Discard consecutive blanks at wrap points, but preserve blanks after linebreaks or at line starts
- **Alignment Handling** — For right-aligned text, properly adjust for blanks preceding linebreaks

> ⚠️ **Note**: These behaviors match standard word processing programs and text editors.

## Cross-Platform Compatibility 🌐

- **Code Portability** — FIGlet is written in C, while FIGWin 1.0 uses Visual Basic
- **Integration Path** — Future FIGWin versions should use a GUI C programming language
- **Plugin Architecture** — FIGlet's code should be modular for easy integration into GUI shells
- **Preferred Development** — New platform implementations should use C or a GUI version of C for maximum portability

## Control Files and Compression 🗜️

- **Command Reservation** — Commands "c", "d", and "s" are permanently reserved in control files
- **Compression Standard** — Use ZIP archives for FIGfonts and control files with renamed extensions
- **Single File Requirement** — Archives must contain only one file (additional files are ignored)
- **Extension Modification** — Change ".zip" to ".flf" or ".flc" as appropriate

```ascii
┌────────────────────────────────────────────────────────┐
│ COMPRESSED FIGFONT REQUIREMENTS:                       │
│                                                        │
│ • Single file per ZIP archive                         │
│ • Rename .zip extension to .flf or .flc               │
│ • Internal filename doesn't matter                    │
│ • Only first file is processed                        │
└────────────────────────────────────────────────────────┘
```

## Capability Comparison Chart 📊

| Feature                             | FIGlet 2.2 | FIGWin 1.0 |
| ----------------------------------- | ---------- | ---------- |
| Interpreting Full_Layout parameter  | Yes        | Yes        |
| Universal smushing                  | Yes        | Yes        |
| Supporting multi-byte input formats | Yes        | No         |
| Processing control files            | Yes        | No         |
| Changing default smushing rules     | Yes        | No         |
| Bundled with GUI font editor        | No         | Yes        |
| Vertical fitting and smushing       | No         | Yes        |

These standards ensure consistent behavior across FIGdrivers while allowing for platform-specific innovations that maintain compatibility with the core FIGfont format.

                 ___________           __               _
                 \_   _____/ ____     |__| ____ ___ __ | |
                  |    __)_ /    \    |  |/  _ <   |  || |
                  |        \   |  \   |  (  <_> )___  | \|
                 /_______  /___|  /\__|  |\____// ____| __
                         \/     \/\______|      \/      \/
