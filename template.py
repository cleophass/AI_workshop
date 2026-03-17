from typing import Any, List, Optional
import boto3
from pydantic import model_validator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class BedrockChatModel(BaseChatModel):
    model_config = {"arbitrary_types_allowed": True}

    model_id: str
    region_name: str = "eu-west-3"
    client: Any = None

    @model_validator(mode="after")
    def init_client(self):
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name
        )
        return self

    @property
    def _llm_type(self) -> str:
        return "bedrock-chat"

    def _convert_messages(self, messages: List):
        converted = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            converted.append({
                "role": role,
                "content": [{"text": msg.content}]
            })

        return converted

    def _generate(
        self,
        messages: List,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:

        bedrock_messages = self._convert_messages(messages)

        response = self.client.converse(
            modelId=self.model_id,
            messages=bedrock_messages
        )

        output_text = response["output"]["message"]["content"][0]["text"]

        message = AIMessage(content=output_text)

        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])
