import json
import os
import re
from typing import Any
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MCP_URL = os.getenv(
    "MCP_URL",
    "https://mcp.progressiosolutions.com/mcp",
)

MCP_PROTOCOL_VERSION = os.getenv(
    "MCP_PROTOCOL_VERSION",
    "2025-06-18",
)

LITELLM_BASE_URL = os.getenv(
    "LITELLM_BASE_URL",
    "http://litellm:4000",
).rstrip("/")

LITELLM_MODEL = os.getenv(
    "LITELLM_MODEL",
    "bookstack-default",
)

LITELLM_API_KEY = os.getenv(
    "LITELLM_API_KEY",
    "",
)

LLM_TIMEOUT = float(
    os.getenv(
        "LLM_TIMEOUT",
        "120",
    )
)

MAX_CONTEXT_CHARS = int(
    os.getenv(
        "MAX_CONTEXT_CHARS",
        "30000",
    )
)

MAX_DOCUMENTS = int(
    os.getenv(
        "MAX_DOCUMENTS",
        "5",
    )
)

MAX_TOKENS = int(
    os.getenv(
        "MAX_TOKENS",
        "800",
    )
)

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0",
    )
)


app = FastAPI(
    title="BookStack AI",
    version="0.3.0",
)


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
    )


class Source(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


def clean_html(html: str) -> str:
    """
    Convert BookStack HTML into clean readable text.
    """

    soup = BeautifulSoup(
        html or "",
        "html.parser",
    )

    for element in soup(
        [
            "script",
            "style",
            "noscript",
        ]
    ):
        element.decompose()

    text = soup.get_text("\n")

    lines: list[str] = []

    for line in text.splitlines():
        line = re.sub(
            r"\s+",
            " ",
            line,
        ).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


class MCPClient:
    """
    Minimal MCP HTTP/SSE client for the BookStack MCP server.
    """

    def __init__(self, url: str):
        self.url = url
        self.session_id: str | None = None
        self.request_id = 0

    async def initialize(self) -> None:
        """
        Initialize an MCP session.
        """

        self.request_id += 1

        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "bookstack-ai",
                    "version": app.version,
                },
            },
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Origin": "https://wiki.progressiosolutions.com",
        }

        async with httpx.AsyncClient(
            timeout=30.0
        ) as client:

            response = await client.post(
                self.url,
                headers=headers,
                json=payload,
            )

        response.raise_for_status()

        session_id = response.headers.get(
            "mcp-session-id"
        )

        if not session_id:
            raise RuntimeError(
                "MCP server did not return a session ID"
            )

        self.session_id = session_id

    @staticmethod
    def _parse_sse(
        response_text: str,
    ) -> dict[str, Any]:
        """
        Parse a JSON-RPC message from an SSE response.
        """

        for line in response_text.splitlines():

            if line.startswith("data: "):

                return json.loads(
                    line[6:]
                )

        raise RuntimeError(
            "MCP response did not contain an SSE data message"
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Call an MCP tool.
        """

        if not self.session_id:
            await self.initialize()

        self.request_id += 1

        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Mcp-Session-Id": self.session_id,
            "Origin": "https://wiki.progressiosolutions.com",
        }

        async with httpx.AsyncClient(
            timeout=60.0
        ) as client:

            response = await client.post(
                self.url,
                headers=headers,
                json=payload,
            )

        response.raise_for_status()

        result = self._parse_sse(
            response.text
        )

        if "error" in result:
            raise RuntimeError(
                str(result["error"])
            )

        mcp_result = result.get(
            "result",
            {},
        )

        if mcp_result.get("isError"):

            content = mcp_result.get(
                "content",
                [],
            )

            messages = [
                item.get(
                    "text",
                    "",
                )
                for item in content
                if item.get("type") == "text"
            ]

            raise RuntimeError(
                "\n".join(messages)
                or "MCP tool returned an error"
            )

        return mcp_result

def extract_search_results(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract BookStack search results from either structuredContent
    or text content.
    """

    structured = result.get(
        "structuredContent"
    )

    if isinstance(
        structured,
        dict,
    ):

        results = structured.get(
            "results",
            [],
        )

        if isinstance(
            results,
            list,
        ):
            return results

    content = result.get(
        "content",
        [],
    )

    if not isinstance(
        content,
        list,
    ):
        return []

    for item in content:

        if item.get("type") != "text":
            continue

        try:

            data = json.loads(
                item["text"]
            )

        except (
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            continue

        results = data.get(
            "results",
            [],
        )

        if isinstance(
            results,
            list,
        ):
            return results

    return []


def extract_page_data(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract BookStack page data from either structuredContent
    or text content.
    """

    structured = result.get(
        "structuredContent"
    )

    if isinstance(
        structured,
        dict,
    ):

        data = structured.get(
            "data",
            {},
        )

        if isinstance(
            data,
            dict,
        ):
            return data

    content = result.get(
        "content",
        [],
    )

    if not isinstance(
        content,
        list,
    ):
        return {}

    for item in content:

        if item.get("type") != "text":
            continue

        try:

            data = json.loads(
                item["text"]
            )

        except (
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            continue

        page_data = data.get(
            "data",
            {},
        )

        if isinstance(
            page_data,
            dict,
        ):
            return page_data

    return {}


async def search_bookstack(
    mcp: MCPClient,
    query: str,
) -> list[dict[str, Any]]:
    """
    Search BookStack using multiple lexical queries.

    This is intentionally kept as the current retrieval mechanism.

    Semantic/vector retrieval can be added later without changing
    the LLM/provider architecture.
    """

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "tell",
        "that",
        "the",
        "their",
        "this",
        "to",
        "us",
        "was",
        "what",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
    }

    words = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        query,
    )

    keywords = [
        word
        for word in words
        if (
            word.lower() not in stop_words
            and len(word) >= 2
        )
    ]

    searches: list[str] = []

    if keywords:

        searches.append(
            " ".join(keywords)
        )

    original_query = query.strip()

    if (
        original_query
        and original_query not in searches
    ):
        searches.append(
            original_query
        )

    for index in range(
        len(keywords) - 1
    ):

        phrase = (
            f"{keywords[index]} "
            f"{keywords[index + 1]}"
        )

        if phrase not in searches:

            searches.append(
                phrase
            )

    for keyword in keywords:

        if keyword not in searches:

            searches.append(
                keyword
            )

    results_by_id: dict[
        int,
        dict[str, Any],
    ] = {}

    for search_query in searches[:8]:

        result = await mcp.call_tool(
            "bookstack_search",
            {
                "query": search_query,
                "count": 10,
            },
        )

        results = extract_search_results(
            result
        )

        for item in results:

            item_id = item.get(
                "id"
            )

            if item_id is not None:

                results_by_id[
                    item_id
                ] = item

    return list(
        results_by_id.values()
    )


async def read_page(
    mcp: MCPClient,
    page_id: int,
) -> dict[str, Any]:
    """
    Read a complete BookStack page.
    """

    result = await mcp.call_tool(
        "bookstack_content_crud",
        {
            "action": "read_page",
            "content_id": page_id,
        },
    )

    return extract_page_data(
        result
    )


def build_context(
    documents: list[dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[
    str,
    list[Source],
]:
    """
    Build LLM context and source list from BookStack documents.
    """

    context_parts: list[str] = []

    sources: list[Source] = []

    seen_urls: set[str] = set()

    current_length = 0

    for document in documents:

        title = (
            document.get("name")
            or document.get("title")
            or "Untitled"
        )

        html = (
            document.get("html")
            or document.get("raw_html")
            or document.get("content")
            or ""
        )

        text = clean_html(
            str(html)
        )

        if not text:
            continue

        url = str(
            document.get("url")
            or ""
        )

        if not url:

            slug = re.sub(
                r"[^a-z0-9]+",
                "-",
                title.lower(),
            ).strip("-")

            url = (
                "https://wiki.progressiosolutions.com/"
                "books/infrastructure-documentation/"
                f"page/{slug}"
            )

        if url not in seen_urls:

            sources.append(
                Source(
                    title=title,
                    url=url,
                )
            )

            seen_urls.add(url)

        part = (
            f"DOCUMENT: {title}\n"
            f"SOURCE: {url}\n\n"
            f"{text}\n"
        )

        if (
            current_length
            + len(part)
            > max_chars
        ):

            remaining = (
                max_chars
                - current_length
            )

            if remaining <= 500:
                break

            part = part[
                :remaining
            ]

        context_parts.append(
            part
        )

        current_length += len(part)

        if current_length >= max_chars:
            break

    return (
        "\n\n---\n\n".join(
            context_parts
        ),
        sources,
    )

SYSTEM_PROMPT = """
You are the AI assistant for a BookStack documentation system.

Your job is to answer questions using ONLY the BookStack documentation
provided in the context.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge to fill missing information.
3. If the documentation does not contain the answer, clearly say that
   the information was not found in the BookStack documentation.
4. Give concise, direct answers.
5. When the documentation contains a specific environment, server,
   service, location, configuration value, command, URL, or other
   concrete detail, state it exactly.
6. Do not mention the internal retrieval process unless necessary.
7. Do not claim that you searched documents that were not provided.
8. Prefer information from the supplied documentation over assumptions.
9. If multiple documents contain conflicting information, clearly state
   the conflict rather than choosing one without evidence.
"""

class LiteLLMClient:
    """
    Provider-neutral LLM client.

    This service communicates with LiteLLM using the OpenAI-compatible
    /v1/chat/completions endpoint.

    The actual provider/model is configured inside LiteLLM.

    Example:

        BookStack AI
            |
            | model=bookstack-default
            v
        LiteLLM
            |
            +--> Azure OpenAI
            +--> OpenAI
            +--> Anthropic
            +--> Gemini
            +--> Ollama
            +--> vLLM
            +--> etc.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
    ):
        self.base_url = base_url.rstrip(
            "/"
        )

        self.model = model

        self.api_key = api_key

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Send a chat completion request to LiteLLM.
        """

        url = (
            f"{self.base_url}"
            "/v1/chat/completions"
        )

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:

            headers[
                "Authorization"
            ] = (
                f"Bearer {self.api_key}"
            )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

        async with httpx.AsyncClient(
            timeout=LLM_TIMEOUT
        ) as client:

            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )

        if response.status_code >= 400:

            detail = response.text[:2000]

            raise RuntimeError(
                "LiteLLM request failed "
                f"with HTTP "
                f"{response.status_code}: "
                f"{detail}"
            )

        try:

            data = response.json()

        except ValueError as exc:

            raise RuntimeError(
                "LiteLLM returned invalid JSON"
            ) from exc

        choices = data.get(
            "choices",
            [],
        )

        if not choices:

            raise RuntimeError(
                "LiteLLM returned no choices"
            )

        message = (
            choices[0]
            .get("message", {})
        )

        answer = (
            message.get(
                "content"
            )
            or ""
        ).strip()

        if not answer:

            raise RuntimeError(
                "LiteLLM returned an empty answer"
            )

        return answer


def create_llm_client() -> LiteLLMClient:
    """
    Create the provider-neutral LLM client.
    """

    return LiteLLMClient(
        base_url=LITELLM_BASE_URL,
        model=LITELLM_MODEL,
        api_key=LITELLM_API_KEY,
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Basic application health check.
    """

    return {
        "status": "ok",
        "service": "bookstack-ai",
        "version": app.version,
        "llm": {
            "provider": "litellm",
            "model": LITELLM_MODEL,
        },
    }


@app.get("/health/llm")
async def health_llm() -> dict[str, Any]:
    """
    Check whether LiteLLM is reachable.

    This does not make an LLM completion request.
    """

    url = (
        f"{LITELLM_BASE_URL}"
        "/v1/models"
    )

    headers = {}

    if LITELLM_API_KEY:

        headers[
            "Authorization"
        ] = (
            f"Bearer {LITELLM_API_KEY}"
        )

    try:

        async with httpx.AsyncClient(
            timeout=10.0
        ) as client:

            response = await client.get(
                url,
                headers=headers,
            )

        response.raise_for_status()

        return {
            "status": "ok",
            "service": "litellm",
            "base_url": LITELLM_BASE_URL,
            "model": LITELLM_MODEL,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "LiteLLM is not reachable: "
                f"{exc}"
            ),
        ) from exc


@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Answer a question using BookStack documentation
    and the provider-neutral LiteLLM gateway.
    """

    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty",
        )

    try:


        mcp = MCPClient(
            MCP_URL
        )

        search_results = (
            await search_bookstack(
                mcp,
                message,
            )
        )

        documents: list[
            dict[str, Any]
        ] = []

        for result in search_results[
            :MAX_DOCUMENTS
        ]:

            page_id = result.get(
                "id"
            )

            if not page_id:
                continue

            try:

                page = await read_page(
                    mcp,
                    int(page_id),
                )

            except Exception:
                continue

            if not page:
                continue


            if (
                not page.get("title")
                and result.get("title")
            ):

                page["title"] = result[
                    "title"
                ]

            if (
                not page.get("name")
                and result.get("title")
            ):

                page["name"] = result[
                    "title"
                ]

            if (
                not page.get("url")
                and result.get("url")
            ):

                page["url"] = result[
                    "url"
                ]

            documents.append(
                page
            )

        if not documents:

            return ChatResponse(
                answer=(
                    "I couldn't find any relevant "
                    "information in the BookStack "
                    "documentation."
                ),
                sources=[],
            )


        context, sources = (
            build_context(
                documents
            )
        )

        if not context.strip():

            return ChatResponse(
                answer=(
                    "I found relevant BookStack "
                    "pages, but they did not contain "
                    "readable documentation."
                ),
                sources=sources,
            )


        llm = create_llm_client()

        user_prompt = (
            "BookStack documentation:\n\n"
            f"{context}\n\n"
            "---\n\n"
            "User question:\n\n"
            f"{message}"
        )

        answer = await llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )


        return ChatResponse(
            answer=answer,
            sources=sources,
        )

    except httpx.HTTPError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to communicate "
                f"with MCP: {exc}"
            ),
        ) from exc

    except RuntimeError as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"AI request failed: {exc}"
            ),
        ) from exc
