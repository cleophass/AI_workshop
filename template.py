from typing import Any, List, Optional, Sequence, Union
import boto3
from pydantic import model_validator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


class BedrockChatModel(BaseChatModel):
    model_config = {"arbitrary_types_allowed": True}

    model_id: str
    region_name: str = "eu-west-3"
    client: Any = None
    tools: Optional[List[dict]] = None

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

    def bind_tools(
        self,
        tools: Sequence[Union[dict, type, BaseTool]],
        **kwargs
    ) -> "BedrockChatModel":
        bedrock_tools = []
        for tool in tools:
            openai_tool = convert_to_openai_tool(tool)["function"]
            bedrock_tools.append({
                "toolSpec": {
                    "name": openai_tool["name"],
                    "description": openai_tool.get("description", ""),
                    "inputSchema": {
                        "json": openai_tool.get("parameters", {"type": "object", "properties": {}})
                    }
                }
            })
        return self.model_copy(update={"tools": bedrock_tools})

    def _convert_messages(self, messages: List):
        converted = []
        system = None

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system = msg.content
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": [{"text": msg.content}]})
            elif isinstance(msg, ToolMessage):
                converted.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": msg.tool_call_id,
                            "content": [{"text": msg.content}]
                        }
                    }]
                })
            else:
                converted.append({"role": "assistant", "content": [{"text": msg.content}]})

        return system, converted

    def _generate(
        self,
        messages: List,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:

        system, bedrock_messages = self._convert_messages(messages)

        params: dict = {"modelId": self.model_id, "messages": bedrock_messages}

        if system:
            params["system"] = [{"text": system}]

        if self.tools:
            params["toolConfig"] = {
                "tools": self.tools,
                "toolChoice": {"any": {}}
            }

        response = self.client.converse(**params)

        content = response["output"]["message"]["content"]
        stop_reason = response.get("stopReason")

        if stop_reason == "tool_use":
            tool_calls = []
            for block in content:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_calls.append({
                        "name": tool_use["name"],
                        "args": tool_use["input"],
                        "id": tool_use["toolUseId"],
                        "type": "tool_call"
                    })
            message = AIMessage(content="", tool_calls=tool_calls)
        else:
            output_text = content[0]["text"]
            message = AIMessage(content=output_text)

        return ChatResult(generations=[ChatGeneration(message=message)])
