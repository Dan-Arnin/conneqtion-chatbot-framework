from enum import Enum
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI


class LLM(Enum):
    OPENAI = "OPENAI"
    MISTRAL = "MISTRAL"
    LLAMA = "LLAMA"
    COHERE = "COHERE"
    CLAUDE = "CLAUDE"


class OPENAILLM:
    def __init__(self, model_variant="gpt-3.5-turbo-0125"):
        self.model = ChatOpenAI(model=model_variant)


class CLAUDE:
    def __init__(self, model_variant="claude-3-sonnet-20240229"):
        self.model = ChatAnthropic(model=model_variant)


class COHERE:
    def __init__(self, model_variant="command-r"):
        self.model = ChatCohere(model=model_variant)


class LLAMA:
    def __init__(self, model_variant="llama3-8b-8192"):
        self.model = ChatGroq(model=model_variant)


class MISTRAL:
    def __init__(self, model_variant="open-mixtral-8x22b"):
        self.model = ChatMistralAI(model=model_variant)


class llm_models:
    llm_class_map = {
        LLM.OPENAI: OPENAILLM,
        LLM.LLAMA: LLAMA,
        LLM.COHERE: COHERE,
        LLM.MISTRAL: MISTRAL,
        LLM.CLAUDE: CLAUDE,
    }

    def __init__(self, model_type: str, model_variant: str):
        try:
            self.model_type = LLM[model_type.upper()]
        except KeyError:
            valid_options = ", ".join([e.value for e in LLM])
            raise ValueError(
                f"Invalid model type '{model_type}'. Valid options are: {valid_options}"
            )
        llm_class = self.llm_class_map.get(self.model_type)
        if llm_class is None:
            raise ValueError("Invalid LLM type")

        self.model = llm_class(model_variant)

    def __repr__(self) -> str:
        return f"llm_models(model_type={self.model_type.value}, model_variant={self.model.model.model})"
