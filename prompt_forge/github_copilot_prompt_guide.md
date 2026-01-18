# How to Write Better Prompts for GitHub Copilot

This guide presents precision techniques for communicating effectively with GitHub Copilot through structured prompt engineering practices.

**GitHub Copilot X optimization framework**
*Eidos*
*March 31, 2025 | 25 minutes*

## Introduction

Generative AI coding tools are transforming developer workflows, accelerating everything from documentation to unit test creation. However, as with any computational system, input quality directly influences output quality. This document addresses the critical patterns for optimizing communication with AI coding assistants.

### 🧪 Case Study: Drawing an Ice Cream Cone 🍦

The difference between receiving optimal versus suboptimal code generation often lies in prompt structure and specificity.

|     Approach     | Input Pattern                                                                                       | Result                                                     |
| :--------------: | :-------------------------------------------------------------------------------------------------- | :--------------------------------------------------------- |
| **❌ Suboptimal** | `Draw an ice cream cone with ice cream using p5.js`                                                 | Generated output resembling a target/bullseye on a stand   |
|  **✅ Optimal**   | `Draw an ice cream cone with an ice cream scoop and a cherry on top` with component specifications: | Correctly generated ice cream cone matching specifications |

1. `The ice cream cone will be a triangle with the point facing down, wider point at the top. It should have light brown fill`
2. `The ice cream scoop will be a half circle on top of the cone with a light pink fill`
3. `The cherry will be a circle on top of the ice cream scoop with a red fill`
4. `Light blue background` | Correctly generated ice cream cone matching specifications |

## Core Framework

This guide systematically addresses:

1. Prompt definition and engineering principles across developer/researcher contexts
2. Three structural patterns for optimizing GitHub Copilot interactions
3. Three supplementary techniques for edge case handling
4. Implementation example for browser extension development

> **Implementation Note:** This document represents patterns derived from systematic analysis rather than comprehensive rules. These techniques are designed for continuous refinement through recursive application.

## Prompt Engineering: Dual Perspectives

In computational contexts, terminology carries different implications depending on system layer interaction.

### Definitional Matrix

| Concept                | Developer Context                                                                            | ML Research Context                                                                                                             |
| :--------------------- | :------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| **Prompts**            | Code blocks, lines, or natural language comments written to generate specific AI suggestions | Algorithm-generated compilation of IDE code and contextual elements continuously sent to foundation models                      |
| **Prompt Engineering** | Strategic comment/instruction design to produce targeted code generation                     | Algorithm development for effective context capture and transformation for LLM consumption                                      |
| **Context**            | Developer-specified parameters defining desired output                                       | Algorithmic extraction of relevant environment data (open files, cursor position, etc.) provided to LLMs as supplementary input |

## Optimal Prompt Patterns

### 1️⃣ Context Initialization

**Application:** When starting with blank files or empty codebases
**Technique:** Establish comprehensive system context before implementation details

```javascript
/*
Create a basic markdown editor in Next.js with the following features:

- Use react hooks
- Create state for markdown with default text "type markdown here"
- A text area where users can write markdown
- Show a live preview of the markdown text as I type
- Support for basic markdown syntax like headers, bold, italics
- Use React markdown npm package
- The markdown text and resulting HTML should be saved in the component's state and updated in real time
*/
```

**Outcome:** Complete functional implementation ready for styling in <30 seconds

> **Calibration Note:** Even with precise initialization, expect probabilistic variance. Example: Despite specifying `"type markdown here"` as default text, the system generated `"markdown preview"` instead.

### 2️⃣ Sequential Decomposition

**Core principle:** Complex operations function better as discrete execution steps
**Implementation:** Generate and validate code incrementally rather than requesting complete solutions

#### Implementation Pattern: Sentence Reversal Function

Sequential prompting with discrete transformations:

1. First, let's make the first letter of the sentence lower case if it's not an 'I'
2. Next, let's split the sentence into an array of words
3. Then, let's take out the punctuation marks from the sentence
4. Now, let's remove the punctuation marks from the sentence
5. Let's reverse the sentence and join it back together
6. Finally, let's make the first letter of the sentence capital and add the punctuation marks

**System advantage:** This pattern enables verification and refinement at each transformation step.

### 3️⃣ Examples as Specification

**Key mechanism:** Examples function as executable specifications for complex data structures
**Application context:** Operations involving nested data or multi-step transformations

#### Data Structure Processing Example

```javascript
const data = [
  [
    { name: 'John', age: 25 },
    { name: 'Jane', age: 30 }
  ],
  [
    { name: 'Bob', age: 40 }
#### Comparative Performance Analysis

| Approach         | Implementation | Result |
| :--------------- | :------------- | :----- |
| **Non-exemplar** | ```javascript  |
// Map through an array of arrays of objects to transform data
const data = [/* ... */];

const mappedData = data.map(x => x.name);
console.log(mappedData);
// Results: [undefined, undefined]

const mappedData = data.map(x => x.name);
console.log(mappedData);
// Results: [undefined, undefined]

``` | ❌ Incorrect transformation |
| **Exemplar-based** | ```javascript
// Map through an array of arrays of objects
// Example: Extract names from the data array
// Desired outcome: ['John', 'Jane', 'Bob']
const data = [...];

const mappedData = data.flatMap(sublist => sublist.map(person => person.name));
console.log(mappedData);
// Results: ['John', 'Jane', 'Bob']
``` | ✅ Correct transformation |

> **Advanced pattern:** Research zero-shot, one-shot, and few-shot learning for deeper understanding of example-based reasoning in LLMs.

## Supplementary Techniques

### 1️⃣ Iterative Refinement

**Principle:** Prompt crafting functions as a convergent optimization process
**Method:** Systematically refine prompts based on output evaluation

#### Progressive Refinement Pattern

| Stage         | Prompt                                                                                                                                                        | Effectiveness           |
| :------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------- |
| **Initial**   | `# Write some code for grades.py`                                                                                                                             | ❌ Under-constrained     |
| **Improved**  | `# Implement a function in grades.py to calculate the average grade`                                                                                          | ⚠️ Partially constrained |
| **Optimized** | `# Implement the function calculate_average_grade in grades.py that takes a list of grades as input and returns the average grade as a floating-point number` | ✅ Fully constrained     |

### 2️⃣ Contextual Environment Optimization

GitHub Copilot utilizes "neighboring tabs" to construct its understanding of the development environment. One or two relevant files typically provide sufficient context.

> **Implementation detail:** The system applies variable weighting to open files; not all contextual elements receive equal consideration.

### 3️⃣ Code Quality Patterns
#### Pattern Comparison

| Implementation Style | Code Input | Generated Output |
| :------------------- | :--------- | :--------------- |
| **Structured**       | ```python  |
def authenticate_user(username, password):
| Implementation Style | Code Input | Generated Output |
| :------------------- | :--------- | :--------------- |
| **Structured**       | ```python  |
def authenticate_user(username, password):
``` | ```python
def authenticate_user(username, password):
    # Code for authenticating the user
    if is_valid_user(username, password):
        generate_session_token(username)
        return True
    else:
        return False
``` |
| **Unstructured** | ```python
def rndpwd(l):
``` | ```python
def rndpwd(l):
    # Code goes here
``` |

## Resources

### GitHub Copilot Documentation

For latest features and functionality:

- Comprehensive implementation guide with practical examples
- GitHub Copilot tag and category on the GitHub Blog
- Official documentation and the GitHub Copilot Chat cookbook

## Security Protocol

**Verification requirement:** AI-generated code requires human validation. LLMs identify and extrapolate patterns from training data, potentially generating novel code sequences. Apply the same rigorous review process to AI-generated code as you would to code from any collaborator.
