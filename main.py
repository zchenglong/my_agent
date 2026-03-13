"""
Content Creation Crew - 内容创作智能体团队

使用 CrewAI 框架构建的内容创作流水线：
研究员 → 作家 → 编辑

支持两种 LLM 接入方式：
1. CrewAI 内置 LLM（工厂模式）
2. 自定义 QwenLLM（继承 BaseLLM）
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from crewai import Agent, Task, Crew, Process, LLM
from crewai.llms.base_llm import BaseLLM
from crewai.events.types.llm_events import LLMCallType
import os

if TYPE_CHECKING:
    from crewai.agent.core import Agent as AgentType
    from crewai.task import Task as TaskType
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
# 第三方库日志压到 WARNING，避免刷屏
for _lib in ("fontTools", "httpx", "httpcore", "urllib3", "fpdf"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

load_dotenv(override=True)


# ── QwenLLM (自定义 BaseLLM 实现) ──────────────────────

logger = logging.getLogger(__name__)


class QwenLLM(BaseLLM):
    """基于 DashScope OpenAI 兼容接口的千问 LLM 实现。"""

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            provider="dashscope",
            **kwargs,
        )
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: TaskType | None = None,
        from_agent: AgentType | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        formatted = self._format_messages(messages)

        self._emit_call_started_event(
            messages=formatted,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
        )

        if from_agent is None:
            if not self._invoke_before_llm_call_hooks(formatted, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

        params: dict[str, Any] = {
            "model": self.model,
            "messages": formatted,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature

        # 构建 OpenAI 格式的 tools 参数
        if tools:
            params["tools"] = self._convert_tools(tools)

        agent_role = getattr(from_agent, "role", None) if from_agent else None
        logger.info("[QwenLLM] 发送请求 → agent=%s, model=%s, messages=%d条, tools=%s",
                     agent_role or "N/A", self.model, len(formatted), len(tools) if tools else 0)
        logger.debug("[QwenLLM] 请求内容: %s", params)

        try:
            response = self._client.chat.completions.create(**params)
        except Exception as e:
            logger.error("[QwenLLM] 请求失败: %s", e)
            self._emit_call_failed_event(
                error=str(e), from_task=from_task, from_agent=from_agent
            )
            raise

        choice = response.choices[0]

        logger.info("[QwenLLM] 收到响应 ← finish_reason=%s, usage=%s",
                     choice.finish_reason,
                     {
                         "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                         "completion_tokens": getattr(response.usage, "completion_tokens", None),
                     } if response.usage else None)
        logger.debug("[QwenLLM] 响应内容: %s", choice.message.content[:500] if choice.message.content else "")

        # 记录 token 用量
        if response.usage:
            self._track_token_usage_internal({
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
            })

        # 处理工具调用
        if choice.message.tool_calls and available_functions:
            tool_call = choice.message.tool_calls[0]
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            result = self._handle_tool_execution(
                function_name=fn_name,
                function_args=fn_args,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )
            return result

        content = choice.message.content or ""
        content = self._apply_stop_words(content)

        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=formatted,
        )

        if from_agent is None and isinstance(content, str):
            content = self._invoke_after_llm_call_hooks(formatted, content, from_agent)

        # 结构化输出
        if response_model:
            return self._validate_structured_output(content, response_model)

        return content

    def supports_stop_words(self) -> bool:
        return self._supports_stop_words_implementation()

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将 CrewAI 工具 schema 转为 OpenAI function calling 格式。"""
        openai_tools = []
        for tool in tools:
            if "function" in tool:
                openai_tools.append(tool)
            else:
                openai_tools.append({"type": "function", "function": tool})
        return openai_tools


# ── LLM 实例 ──────────────────────────────────────────

_api_key = os.getenv("DASHSCOPE_API_KEY")
_base_url = os.getenv("DASHSCOPE_API_BASE_URL")

if not _api_key:
    raise SystemExit(
        "错误: 未设置 DASHSCOPE_API_KEY。\n"
        "请在 .env 文件中添加，或通过环境变量导出:\n"
        "  export DASHSCOPE_API_KEY=sk-xxx\n"
        "  export DASHSCOPE_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

# 方式一：CrewAI 内置 LLM（工厂模式）
default_llm = LLM(
    model="qwen-plus",
    api_key=_api_key,
    base_url=_base_url,
)

# 方式二：自定义 QwenLLM（继承 BaseLLM）
qwen_llm = QwenLLM(
    model="qwen-plus",
    api_key=_api_key,
    base_url=_base_url,
)


# ── Agents ──────────────────────────────────────────────

def build_agents(llm: BaseLLM) -> tuple[Agent, Agent, Agent, Agent]:
    """构建四个内容创作 Agent。"""

    researcher = Agent(
        role="资深内容研究员",
        goal="围绕给定主题，搜集全面、准确、有深度的素材和观点",
        backstory=(
            "你是一位经验丰富的研究员，擅长快速定位关键信息，"
            "能够从多角度梳理一个话题的核心要点、最新进展和独到见解。"
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="资深内容作家",
        goal="根据研究素材，撰写结构清晰、引人入胜的高质量文章",
        backstory=(
            "你是一位出色的作家，擅长将复杂信息转化为通俗易懂、"
            "逻辑清晰的文章，文笔流畅且富有感染力。"
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    editor = Agent(
        role="资深内容编辑",
        goal="审校文章质量，确保内容准确、结构合理、语言精炼",
        backstory=(
            "你是一位严谨的编辑，拥有多年内容审校经验，"
            "对文字质量和逻辑严密性有极高标准。"
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    illustrator = Agent(
        role="资深插画策划师",
        goal="为文章挑选 2-3 个最适合配图的位置，设计配图描述，用于 AI 生成插画",
        backstory=(
            "你是一位视觉创意专家，擅长将文字内容转化为生动的画面描述，"
            "善于在整篇文章中挑选最具视觉冲击力的关键节点，设计出具有表现力的插画方案。"
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, editor, illustrator


# ── Tasks ───────────────────────────────────────────────

def build_tasks(
    topic: str, researcher: Agent, writer: Agent, editor: Agent, illustrator: Agent,
) -> list[Task]:
    """根据主题构建任务流水线。"""

    research_task = Task(
        description=(
            f"针对主题「{topic}」进行深入研究。\n"
            "要求：\n"
            "1. 梳理该主题的背景和核心概念\n"
            "2. 总结最新的关键进展和趋势\n"
            "3. 列出 3-5 个独到观点或有价值的数据\n"
            "4. 输出结构化的研究笔记"
        ),
        expected_output="一份结构化的研究笔记，包含背景、关键进展、核心观点和数据",
        agent=researcher,
    )

    write_task = Task(
        description=(
            f"基于研究笔记，撰写一篇关于「{topic}」的文章。\n"
            "要求：\n"
            "1. 文章 1000-5000 字\n"
            "2. 包含引言、正文（3-5 个小节）、结语\n"
            "3. 语言生动，逻辑清晰\n"
            "4. 适当引用研究中的数据和观点"
        ),
        expected_output="一篇 1000-5000 字的完整文章，结构完整，内容充实",
        agent=writer,
    )

    edit_task = Task(
        description=(
            "审校上一步生成的文章。\n"
            "要求：\n"
            "1. 检查事实准确性和逻辑连贯性\n"
            "2. 优化语言表达，去除冗余\n"
            "3. 确保段落过渡自然\n"
            "4. 输出最终定稿版本"
        ),
        expected_output="经过审校优化的最终定稿文章",
        agent=editor,
    )

    illustrate_task = Task(
        description=(
            "分析上一步的定稿文章，从中挑选 2-3 个最适合配图的位置，设计配图描述。\n"
            "要求：\n"
            "1. 仔细阅读文章，理解整体结构和内容\n"
            "2. 从全文中挑选 2-3 个最具视觉表现力的关键位置（不必每个小节都配图）\n"
            "3. 为每个位置写一句简洁、具象的中文图片描述（适合 AI 绘画生成）\n"
            "4. 描述应包含具体的画面元素、风格和氛围\n"
            "5. 严格按以下格式输出，每行一个：\n"
            "   [IMG: 图片描述内容]\n"
            "6. 在每个 [IMG: ...] 标记前注明对应的小节标题\n"
            "7. 最后完整输出原文，并在选定位置插入对应的 [IMG: ...] 标记\n"
            "8. 注意：总共只需要 2-3 张配图，不要超过 3 张"
        ),
        expected_output=(
            "包含 2-3 个 [IMG: ...] 配图标记的完整文章"
        ),
        agent=illustrator,
    )

    return [research_task, write_task, edit_task, illustrate_task]


# ── Crew ────────────────────────────────────────────────

def create_crew(topic: str, llm: BaseLLM) -> Crew:
    """创建内容创作 Crew。"""
    researcher, writer, editor, illustrator = build_agents(llm)
    return Crew(
        agents=[researcher, writer, editor, illustrator],
        tasks=build_tasks(topic, researcher, writer, editor, illustrator),
        process=Process.sequential,
        verbose=True,
    )


# ── PDF 输出 ──────────────────────────────────────────

_FONT_PATHS = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _find_chinese_font() -> str:
    """返回系统中第一个可用的中文字体路径，找不到则抛异常。"""
    for path in _FONT_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "未找到可用的中文字体文件，请手动指定字体路径。\n"
        f"已搜索: {_FONT_PATHS}"
    )


def _output_dir() -> str:
    """返回 output 目录路径，不存在则自动创建。"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _safe_filename(topic: str) -> str:
    """根据主题生成安全的文件名（不含扩展名）。"""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in topic)[:50]


# ── 图片生成 ──────────────────────────────────────────

_IMG_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
_IMG_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"
_IMG_MODEL = "wanx2.1-t2i-turbo"


def generate_image(prompt: str, index: int = 0) -> str | None:
    """调用 DashScope 通义万相 API 生成图片，返回本地文件路径。失败返回 None。"""
    import requests

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.warning("[图片生成] 未设置 DASHSCOPE_API_KEY，跳过图片生成")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload = {
        "model": _IMG_MODEL,
        "input": {"prompt": prompt},
        "parameters": {"n": 1, "size": "1024*1024"},
    }

    logger.info("[图片生成] 提交任务: %s", prompt[:80])
    try:
        resp = requests.post(_IMG_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        task_id = resp.json()["output"]["task_id"]
    except Exception as e:
        logger.error("[图片生成] 提交失败: %s", e)
        return None

    # 轮询等待结果
    poll_headers = {"Authorization": f"Bearer {api_key}"}
    for _ in range(60):  # 最多等待 3 分钟
        time.sleep(3)
        try:
            status_resp = requests.get(
                f"{_IMG_TASK_URL}/{task_id}", headers=poll_headers, timeout=15
            )
            status_resp.raise_for_status()
            data = status_resp.json()
            task_status = data["output"]["task_status"]
        except Exception as e:
            logger.error("[图片生成] 轮询失败: %s", e)
            return None

        if task_status == "SUCCEEDED":
            img_url = data["output"]["results"][0]["url"]
            # 下载图片
            try:
                img_resp = requests.get(img_url, timeout=30)
                img_resp.raise_for_status()
                filepath = os.path.join(_output_dir(), f"img_{index}.png")
                with open(filepath, "wb") as f:
                    f.write(img_resp.content)
                logger.info("[图片生成] 已保存: %s", filepath)
                return filepath
            except Exception as e:
                logger.error("[图片生成] 下载失败: %s", e)
                return None
        elif task_status == "FAILED":
            logger.error("[图片生成] 任务失败: %s", data)
            return None
        else:
            logger.debug("[图片生成] 状态: %s", task_status)

    logger.error("[图片生成] 超时")
    return None


def _translate_to_keywords(description: str) -> str:
    """用 Qwen 将中文图片描述翻译为英文搜索关键词。"""
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_API_BASE_URL"),
        )
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{
                "role": "user",
                "content": (
                    "请把以下中文图片描述提炼为 3-5 个英文搜索关键词，"
                    "用英文逗号分隔，只输出关键词，不要其他内容：\n"
                    f"{description}"
                ),
            }],
        )
        keywords = resp.choices[0].message.content.strip()
        logger.info("[关键词翻译] %s → %s", description[:30], keywords)
        return keywords
    except Exception as e:
        logger.warning("[关键词翻译] 失败: %s", e)
        return "technology, illustration"


def search_image(description: str, index: int = 0) -> str | None:
    """搜索网络图片，返回本地文件路径。失败返回 None。

    优先使用 Pexels API（需在 .env 中配置 PEXELS_API_KEY），
    否则回退到 Bing 图片搜索抓取。
    """
    import re
    import requests
    from urllib.parse import quote_plus

    keywords = _translate_to_keywords(description)
    filepath = os.path.join(_output_dir(), f"img_{index}.png")

    # 方式一：Pexels API（有 key 时使用）
    pexels_key = os.getenv("PEXELS_API_KEY")
    if pexels_key:
        logger.info("[网络搜图/Pexels] 搜索: %s", keywords)
        try:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": pexels_key},
                params={"query": keywords, "per_page": 1, "size": "large"},
                timeout=15,
            )
            resp.raise_for_status()
            photos = resp.json().get("photos", [])
            if photos:
                img_url = photos[0]["src"]["large"]
                img_resp = requests.get(img_url, timeout=15)
                img_resp.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(img_resp.content)
                logger.info("[网络搜图/Pexels] 已保存: %s", filepath)
                return filepath
        except Exception as e:
            logger.warning("[网络搜图/Pexels] 失败: %s", e)

    # 方式二：Bing 图片搜索抓取（无需 key）
    logger.info("[网络搜图/Bing] 搜索: %s", keywords)
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        search_url = f"https://www.bing.com/images/search?q={quote_plus(keywords)}&first=1"
        resp = requests.get(search_url, headers=headers, timeout=15)
        resp.raise_for_status()
        # 提取 murl（原图地址）
        murls = re.findall(r'murl&quot;:&quot;(https?://[^&]+?)&quot;', resp.text)
        for img_url in murls[:5]:  # 尝试前 5 个
            try:
                img_resp = requests.get(img_url, headers=headers, timeout=10)
                if img_resp.status_code == 200 and len(img_resp.content) > 5000:
                    with open(filepath, "wb") as f:
                        f.write(img_resp.content)
                    logger.info("[网络搜图/Bing] 已保存: %s", filepath)
                    return filepath
            except Exception:
                continue
        logger.warning("[网络搜图/Bing] 未找到可下载的图片")
    except Exception as e:
        logger.error("[网络搜图/Bing] 失败: %s", e)

    return None


def save_as_pdf(content: str, topic: str, images: dict[str, str] | None = None) -> str:
    """将文章内容保存为 PDF 文件，返回文件路径。

    Args:
        content: 文章 Markdown 文本（可含 [IMG: ...] 标记）
        topic: 文章主题（用于生成文件名）
        images: 图片描述 → 本地图片路径的映射
    """
    import re
    from fpdf import FPDF

    # 若未传入 images 映射，自动按出现顺序匹配 output/img_{index}.png
    if not images:
        images = {}
        img_descs = re.findall(r"[\[【]IMG:\s*(.+?)[\]】]", content)
        for i, desc in enumerate(img_descs):
            candidate = os.path.join(_output_dir(), f"img_{i}.png")
            if os.path.isfile(candidate):
                images[desc] = candidate
    else:
        images = dict(images)
    font_path = _find_chinese_font()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.add_font("Chinese", "", font_path)
    pdf.add_font("Chinese", "B", font_path)

    effective_w = pdf.w - pdf.l_margin - pdf.r_margin

    def _strip_md_inline(text: str) -> str:
        """去除行内 Markdown 标记（加粗、斜体、行内代码、链接）。"""
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold**
        text = re.sub(r"__(.+?)__", r"\1", text)       # __bold__
        text = re.sub(r"\*(.+?)\*", r"\1", text)       # *italic*
        text = re.sub(r"_(.+?)_", r"\1", text)         # _italic_
        text = re.sub(r"`(.+?)`", r"\1", text)         # `code`
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) → text
        return text

    def _is_heading(line: str) -> tuple[int, str] | None:
        """识别标题行，返回 (级别, 标题文本) 或 None。"""
        m = re.match(r"^(#{1,3})\s+(.+)$", line)
        if m:
            return len(m.group(1)), _strip_md_inline(m.group(2))
        m = re.match(r"^【(.+?)】(.*)$", line)
        if m:
            extra = m.group(2).strip()
            title = m.group(1) + ("  " + extra if extra else "")
            return 1, _strip_md_inline(title)
        m = re.match(r"^\*\*(.+?)\*\*\s*$", line)
        if m:
            return 2, m.group(1)
        return None

    def _write_heading(level: int, text: str) -> None:
        size = {1: 20, 2: 16, 3: 14}.get(level, 14)
        spacing = {1: 4, 2: 3, 3: 2}.get(level, 2)
        if level == 1:
            pdf.ln(4)
        pdf.set_font("Chinese", "B", size)
        pdf.multi_cell(effective_w, size * 0.55, text)
        pdf.ln(spacing)

    def _insert_image(desc: str) -> None:
        """插入与描述匹配的图片。"""
        img_path = images.get(desc)
        if not img_path or not os.path.isfile(img_path):
            return
        img_w = effective_w * 0.75
        x = pdf.l_margin + (effective_w - img_w) / 2
        pdf.ln(4)
        pdf.image(img_path, x=x, w=img_w)
        pdf.ln(4)

    def _is_subtitle_annotation(text: str) -> bool:
        """识别括号包裹的副标题/注释行（如「（审校定稿｜2024年7月）」）。"""
        return bool(re.match(r"^[（\(].+[）\)]$", text))

    def _is_list_item(text: str) -> tuple[str, str] | None:
        """识别无序列表（- / * ）和有序列表（1. ），返回 (marker, content) 或 None。"""
        m = re.match(r"^([-*])\s+(.+)$", text)
        if m:
            return "•", m.group(2)
        m = re.match(r"^(\d+\.)\s+(.+)$", text)
        if m:
            return m.group(1), m.group(2)
        return None

    # 预处理：将 Markdown 行尾换行（两个空格+换行）替换为真实换行
    content = re.sub(r"  \n", "\n", content)

    for line in content.split("\n"):
        stripped = line.strip()

        # 空行
        if not stripped:
            pdf.ln(4)
            continue

        # [IMG: 描述] 或 【IMG: 描述】 标记 → 插入图片（无图片时跳过）
        img_match = re.match(r"^[\[【]IMG:\s*(.+?)[\]】]\s*$", stripped)
        if img_match:
            desc = img_match.group(1)
            img_path = images.get(desc)
            if img_path and os.path.isfile(img_path):
                _insert_image(desc)
            continue

        # 分隔线 --- / *** / ___
        if re.match(r"^[-*_]{3,}\s*$", stripped):
            pdf.ln(3)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.l_margin + effective_w, y)
            pdf.ln(3)
            continue

        # 标题
        heading = _is_heading(stripped)
        if heading:
            _write_heading(*heading)
            continue

        # 副标题/注释行（括号包裹）
        if _is_subtitle_annotation(stripped):
            text = _strip_md_inline(stripped)
            pdf.set_font("Chinese", "", 10)
            pdf.set_text_color(128, 128, 128)
            pdf.multi_cell(effective_w, 6, text)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            continue

        # 列表项
        list_item = _is_list_item(stripped)
        if list_item:
            marker, item_text = list_item
            text = _strip_md_inline(item_text)
            pdf.set_font("Chinese", "", 12)
            indent = 8
            pdf.set_x(pdf.l_margin + indent)
            pdf.multi_cell(effective_w - indent, 7, f"{marker}  {text}")
            pdf.ln(2)
            continue

        # 普通段落：去除行内 Markdown 标记
        text = _strip_md_inline(stripped)
        pdf.set_font("Chinese", "", 12)
        pdf.multi_cell(effective_w, 7, text)
        pdf.ln(3)  # 段间距

    filepath = os.path.join(_output_dir(), f"{_safe_filename(topic)}.pdf")
    pdf.output(filepath)
    return filepath


def save_as_markdown(content: str, topic: str, images: dict[str, str] | None = None) -> str:
    """将文章内容保存为 Markdown 文件，返回文件路径。

    有图片时将 [IMG: 描述] 替换为 ![描述](路径)，无图片时去掉标记。
    """
    import re

    def _replace_img(m: re.Match) -> str:
        desc = m.group(1)
        if images:
            img_path = images.get(desc)
            if img_path and os.path.isfile(img_path):
                return f"![{desc}]({img_path})"
        return ""

    content = re.sub(r"[\[【]IMG:\s*(.+?)[\]】]", _replace_img, content)
    filepath = os.path.join(_output_dir(), f"{_safe_filename(topic)}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


# ── Main ────────────────────────────────────────────────

def choose_llm() -> BaseLLM:
    """交互式选择 LLM 实现方式。"""
    print("\n可选 LLM 实现:")
    print("  1. CrewAI 内置 LLM（默认）")
    print("  2. 自定义 QwenLLM（继承 BaseLLM）")
    choice = input("请选择 [1/2]（默认 1）: ").strip()
    if choice == "2":
        print("→ 使用自定义 QwenLLM")
        return qwen_llm
    print("→ 使用 CrewAI 内置 LLM")
    return default_llm


def choose_image_source() -> str | None:
    """交互式选择是否生成配图及图片来源。默认不生成。"""
    print("\n是否为文章生成配图?")
    print("  0. 不生成配图（默认）")
    print("  1. AI 生成优先，失败后网络搜图")
    print("  2. 网络搜图优先，失败后 AI 生成")
    print("  3. 仅 AI 生成")
    print("  4. 仅网络搜图")
    choice = input("请选择 [0/1/2/3/4]（默认 0）: ").strip()
    if choice in ("1", "2", "3", "4"):
        labels = {"1": "AI 生成优先", "2": "网络搜图优先", "3": "仅 AI 生成", "4": "仅网络搜图"}
        print(f"→ {labels[choice]}")
        return choice
    print("→ 不生成配图")
    return None


def fetch_image(prompt: str, index: int, mode: str) -> str | None:
    """根据模式获取图片，返回本地路径或 None。"""
    if mode == "1":  # AI 优先
        path = generate_image(prompt, index=index)
        if not path:
            path = search_image(prompt, index=index)
        return path
    elif mode == "2":  # 网络优先
        path = search_image(prompt, index=index)
        if not path:
            path = generate_image(prompt, index=index)
        return path
    elif mode == "3":  # 仅 AI
        return generate_image(prompt, index=index)
    elif mode == "4":  # 仅网络
        return search_image(prompt, index=index)
    return None


if __name__ == "__main__":
    import re as _re

    selected_llm = choose_llm()

    topic = input("请输入文章主题: ").strip()
    if not topic:
        topic = "人工智能在日常生活中的应用"
        print(f"未输入主题，使用默认主题: {topic}")

    crew = create_crew(topic, selected_llm)
    result = crew.kickoff()
    article = str(result)

    print("\n" + "=" * 60)
    print("最终文章")
    print("=" * 60)
    print(article)

    # 解析 [IMG: ...] 标记并生成配图
    img_prompts = _re.findall(r"[\[【]IMG:\s*(.+?)[\]】]", article)
    images: dict[str, str] = {}
    if img_prompts:
        img_mode = choose_image_source()
        if img_mode:
            print(f"\n发现 {len(img_prompts)} 个配图标记，开始获取图片...")
            for i, prompt in enumerate(img_prompts):
                print(f"  [{i + 1}/{len(img_prompts)}] {prompt[:60]}...")
                path = fetch_image(prompt, i, img_mode)
                if path:
                    images[prompt] = path
                    print(f"    ✓ {path}")
                else:
                    print(f"    ✗ 获取失败，跳过")
            print(f"图片获取完成: {len(images)}/{len(img_prompts)} 张成功")

    try:
        md_path = save_as_markdown(article, topic, images=images)
        print(f"\nMarkdown 已保存至: {md_path}")
    except Exception as e:
        print(f"\nMarkdown文件保存失败: {e}")

    try:
        pdf_path = save_as_pdf(article, topic, images=images)
        print(f"PDF 已保存至: {pdf_path}")
    except Exception as e:
        print(f"\nPDF文件保存失败: {e}")


