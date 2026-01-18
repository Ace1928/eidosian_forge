# ⚛️ **sms_forge** v3.14.15 ⚡

> _"Conversation isn't information transfer—it's recursive meaning negotiation across minds."_

Core component for conversational intelligence within the Eidosian Forge ecosystem—where messaging meets memory and intent tracking transforms ephemeral exchanges into structured knowledge.

[![Forge System](https://img.shields.io/badge/Forge-System-8A2BE2)](https://github.com/Ace1928) [![Version](https://img.shields.io/badge/Version-3.14.15-blue)] [![Python](https://img.shields.io/badge/Python-3.12+-purple)](https://www.python.org/) [![License](https://img.shields.io/badge/License-Eidosian-green)](https://github.com/Ace1928/eidosian_forge)

```ascii
╭──────────────────────────────────────────────────────────╮
│ ⊢⊣  MESSAGE → MEANING → MEMORY → MODEL → MANIFESTATION   │
╰──────────────────────────────────────────────────────────╯
```

## 🧠 **Cognitive Foundation** 🌀

The `sms_forge` transforms messaging from transient exchanges into structured knowledge through conversational memory and intent recognition. Unlike standard messaging systems, we don't merely transmit text—we build progressive mental models across interactions, preserve contextual continuity, and track evolving intentions.

```ascii
┌────────────────────────────────────────────────────────────┐
│ STANDARD MESSAGING:           EIDOSIAN MESSAGING:          │
│                                                            │
│ Text → Transmission →         Text → Intent Extraction →   │
│ Ephemeral Exchange            Memory Integration →         │
│                               Contextual Continuity        │
└────────────────────────────────────────────────────────────┘
```

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Messages without memory are like neurons without connections ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

## 💎 **Core Capabilities** 🎯

### **1. Conversational Memory Architecture**

- **Episodic-Semantic Fusion** — Combines narrative memory with structured knowledge extraction
- **Memory Decay Modeling** — Implements cognitively realistic importance-based retention
- **Cross-Conversation Continuity** — Maintains thematic threads across temporal gaps
- **Salience Detection** — Identifies and preserves critical information with precision

### **2. Intent Recognition & Tracking**

- **Multi-Message Intent Detection** — Identifies goals across interaction sequences
- **Intent Evolution Tracking** — Maps how objectives transform through conversation
- **Confidence Calibration** — Quantifies certainty about detected intents with mathematical precision
- **Context-Sensitive Disambiguation** — Resolves unclear signals through multifaceted context

### **3. Platform Integration Framework**

- **Multi-Channel Coherence** — Maintains unified conversational models across platforms
- **Protocol Adaptation Layer** — Normalizes diverse messaging formats into canonical representations
- **Identity Resolution** — Tracks conversational partners across identification changes
- **Cross-Platform Synchronization** — Ensures consistent state across multiple endpoints

```python
def process_message(
    message: Message,
    memory: ConversationalMemory,
    intent_tracker: IntentTracker
) -> ProcessingResult:
    """Process incoming message with memory integration and intent tracking.

    Args:
        message: The incoming message to process
        memory: Existing conversational memory store
        intent_tracker: System for tracking evolving intentions

    Returns:
        Processing results with memory updates and detected intents
    """
    # Extract semantic meaning beyond surface text
    semantic_content = extract_meaning(message.text, message.context)

    # Update our memory graph with new information
    memory_updates = memory.integrate(
        semantic_content,
        importance=calculate_salience(semantic_content)
    )

    # Track how intentions evolve through conversation
    intent_updates = intent_tracker.update(
        semantic_content,
        conversation_history=memory.recent_exchanges(limit=5)
    )

    # Formulate response based on complete understanding
    response = generate_response(
        semantic_content,
        memory=memory,
        detected_intent=intent_updates.current_intent,
        confidence=intent_updates.confidence
    )

    return ProcessingResult(
        response=response,
        memory_updates=memory_updates,
        intent_updates=intent_updates
    )
```

## 🌠 **Integration Architecture** 🧩

The `sms_forge` interfaces seamlessly with the entire Eidosian ecosystem:

- **With `knowledge_forge`**: Enriches conversations with structured knowledge access
- **With `memory_forge`**: Provides specialized episodic memory for conversation sequences
- **With `llm_forge`**: Leverages language models for response generation and intent detection
- **With `type_forge`**: Ensures consistent message schema and validation
- **With `terminal_forge`**: Delivers terminal-based messaging interfaces

```ascii
╭────────────────────╮       ╭────────────────────╮
│  memory_forge      │<─────>│     sms_forge      │
╰────────────────────╯       ╰────────────────────╯
         ▲                           ▲
         │                           │
         ▼                           ▼
╭────────────────────╮       ╭────────────────────╮
│  knowledge_forge   │<─────>│    llm_forge       │
╰────────────────────╯       ╰────────────────────╯
         ▲                           ▲
         │                           │
         ▼                           ▼
╭────────────────────╮       ╭────────────────────╮
│   terminal_forge   │<─────>│ integration APIs   │
╰────────────────────╯       ╰────────────────────╯
```

## 🔍 **Implementation Details** ⚙️

### **Conversational Domain Model**

The system implements a precise domain model for modeling conversations:

```typescript
// Core message representation with rich metadata
interface Message {
  readonly id: string;
  readonly content: string;
  readonly sender: Identity;
  readonly timestamp: DateTime;
  readonly channel: Channel;
  readonly metadata: MessageMetadata;

  // Semantic extraction and relations
  extractEntities(): Entity[];
  getSemanticEmbedding(): Vector;
}

// Memory structure for conversation tracking
interface ConversationalMemory {
  // Integrate new information with importance weighting
  integrate(message: Message, importance: number): MemoryUpdate;

  // Retrieve relevant context by similarity or recency
  findRelevant(query: string, limit?: number): Message[];
  getRecentExchanges(limit: number): Message[];

  // Memory maintenance operations
  consolidate(): void; // Merge related memories
  prune(threshold: number): void; // Remove low-importance items
}
```

### **Architecture Highlights**

- **Memory Graph**: Non-linear conversation representation through semantic connections
- **Intent Recognition**: Multi-layer classification with confidence metrics
- **Platform Adapters**: Protocol-specific modules for diverse messaging platforms
- **Response Generation**: Context-aware text generation with intent alignment
- **Memory Persistence**: Efficient storage with privacy-preserving encryption

## 📊 **Usage Examples** 🔬

### **1. Basic Messaging Flow**

```python
from sms_forge import Conversation, MemoryStore, IntentSystem
from sms_forge.types import Message, Identity, ProcessingOptions

# Initialize conversational components
memory_store = MemoryStore()
intent_system = IntentSystem()
conversation = Conversation(memory=memory_store, intent_tracker=intent_system)

# Process an incoming message
incoming_message = Message(
    content="Could you check if flights to Tokyo are available next week?",
    sender=Identity(id="user123", name="Alice"),
    channel="web"
)

# Process with full cognitive architecture
result = conversation.process_message(
    message=incoming_message,
    options=ProcessingOptions(
        memory_importance=0.8,  # Higher importance for travel requests
        intent_sensitivity=0.7,  # Balanced intent detection threshold
        response_generation=True  # Generate a response automatically
    )
)

print(f"Detected intent: {result.intent.name}, confidence: {result.intent.confidence}")
print(f"Response: {result.response.content}")
print(f"Memory entries created: {len(result.memory_updates)}")
```

### **2. Multi-Platform Integration**

```python
from sms_forge import MessagingHub, PlatformAdapter
from sms_forge.platforms import TelegramAdapter, DiscordAdapter, WebAdapter
from sms_forge.types import ConversationState

# Create platform-specific adapters
telegram = TelegramAdapter(api_token="your_telegram_token")
discord = DiscordAdapter(bot_token="your_discord_token")
web = WebAdapter(endpoint="https://api.example.com/chat")

# Create a unified messaging hub with cross-platform memory
hub = MessagingHub()
hub.register_platform(telegram)
hub.register_platform(discord)
hub.register_platform(web)

# Conversation state synchronized across all platforms
state = ConversationState(user_id="user123")

# Messages from any platform update the unified state
telegram_msg = telegram.receive_message()
hub.process_incoming(telegram_msg, state)

discord_msg = discord.receive_message()
hub.process_incoming(discord_msg, state)  # State maintains continuity

# Respond on appropriate platform
response = hub.generate_response(state)
hub.send_response(response, state.last_platform)
```

## 🔧 **Installation & Setup** 💻

```bash
# Install from PyPI with guaranteed memory integrity
pip install sms-forge==3.14.15

# Verify installation with memory tests
python -m sms_forge.verify_memory

# Or install from source
git clone https://github.com/eidosian/sms_forge.git
cd sms_forge
pip install -e .
```

## 🚀 **Platform Configuration** 🖥️

```bash
# Initialize configuration for messaging platforms
sms-forge init-config

# Connect to Telegram
sms-forge connect telegram --token YOUR_TOKEN

# Connect to Discord
sms-forge connect discord --token YOUR_TOKEN

# Start the messaging hub with all connected platforms
sms-forge start-hub --memory-persistence sqlite
```

## 🤝 **Contribution Guidelines** 🌱

Contributions to `sms_forge` must adhere to Eidosian principles:

- **Memory Integrity** — Conversational state must maintain perfect fidelity
- **Intent Precision** — Intent detection algorithms must be extensively validated
- **Cross-Platform Compatibility** — All platforms must receive equal capability support
- **Privacy Preservation** — User data must be handled with explicit consent tracking
- **Elegant Degradation** — Systems must function gracefully under constrained conditions

```ascii
╭──────────────────────────────────────────────────────────────────╮
│ (￣ ω ￣) "In sms_forge, we don't just remember messages;        │
│           we understand conversations as they unfold in time."    │
╰──────────────────────────────────────────────────────────────────╯
```

## 📚 **Further Resources** 📖

- [Conversation Modeling Guide](https://github.com/Ace1928/sms_forge/docs/conversation.md) - Memory architecture details
- [Intent Recognition Tutorial](https://github.com/Ace1928/sms_forge/docs/intent.md) - Understanding intent detection
- [Platform Integration](https://github.com/Ace1928/sms_forge/docs/platforms.md) - Adding new messaging platforms
- [Memory Persistence Options](https://github.com/Ace1928/sms_forge/docs/persistence.md) - Storage configuration

---

Maintained with recursive precision by Lloyd Handyside <ace1928@gmail.com>
© 3.14.15 - The irrational version for rational minds

> "A conversation without memory is like a database without indexes—technically functional but practically useless." — Eidos
