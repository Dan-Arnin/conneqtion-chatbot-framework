from enum import Enum, auto
from typing import Optional
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory

class MemoryType(Enum):
    ConversationBufferMemory = auto()
    ConversationBufferWindowMemory = auto()
    ConversationSummaryMemory = auto()
    ConversationSummaryBufferMemory = auto()

class ConversationBufferMemory:
    def __init__(self) -> None:
        self.memory = ConversationBufferMemory()

class ConversationBufferWindowMemory:
    def __init__(self, k):
        self.k = k
        self.memory = ConversationBufferWindowMemory(k=self.k)

class ConversationSummaryBufferMemory:
    def __init__(self, token_limit: None, llm : None) -> None:
        self.memory = ConversationSummaryBufferMemory(max_token_limit= token_limit, llm = llm)

class memory:
    memory_class_map = {
        MemoryType.ConversationBufferMemory: ConversationBufferMemory,
        MemoryType.ConversationBufferWindowMemory: ConversationBufferWindowMemory,
        MemoryType.ConversationSummaryBufferMemory: ConversationSummaryBufferMemory,
    }
    def __init__(self, memory_type: str) -> None:
        try:
            self.memory_type = MemoryType[memory_type.upper()]
        except KeyError:
            valid_options = ", ".join([e.value for e in MemoryType])
            raise ValueError(
                f"Invalid model type '{memory_type}'. Valid options are: {valid_options}"
            )
        memory_class = self.memory_class_map.get(self.memory_type)
        if memory_class is None:
            raise ValueError(f"Invalid memory type '{memory_type}'")
        self.memory = memory_class()