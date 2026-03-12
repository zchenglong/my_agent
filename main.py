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

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

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

def build_agents(llm: BaseLLM) -> tuple[Agent, Agent, Agent]:
    """构建三个内容创作 Agent。"""

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

    return researcher, writer, editor


# ── Tasks ───────────────────────────────────────────────

def build_tasks(topic: str, researcher: Agent, writer: Agent, editor: Agent) -> list[Task]:
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
            "1. 文章 800-1200 字\n"
            "2. 包含引言、正文（2-3 个小节）、结语\n"
            "3. 语言生动，逻辑清晰\n"
            "4. 适当引用研究中的数据和观点"
        ),
        expected_output="一篇 800-1200 字的完整文章，结构完整，内容充实",
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

    return [research_task, write_task, edit_task]


# ── Crew ────────────────────────────────────────────────

def create_crew(topic: str, llm: BaseLLM) -> Crew:
    """创建内容创作 Crew。"""
    researcher, writer, editor = build_agents(llm)
    return Crew(
        agents=[researcher, writer, editor],
        tasks=build_tasks(topic, researcher, writer, editor),
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


def save_as_pdf(content: str, topic: str) -> str:
    """将文章内容保存为 PDF 文件，返回文件路径。"""
    import re
    from fpdf import FPDF

    font_path = _find_chinese_font()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.add_font("Chinese", "", font_path)
    pdf.add_font("Chinese", "B", font_path)

    effective_w = pdf.w - pdf.l_margin - pdf.r_margin

    def _strip_md_inline(text: str) -> str:
        """去除行内 Markdown 标记（加粗、斜体、行内代码）。"""
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold**
        text = re.sub(r"__(.+?)__", r"\1", text)       # __bold__
        text = re.sub(r"\*(.+?)\*", r"\1", text)       # *italic*
        text = re.sub(r"_(.+?)_", r"\1", text)         # _italic_
        text = re.sub(r"`(.+?)`", r"\1", text)         # `code`
        return text

    def _is_heading(line: str) -> tuple[int, str] | None:
        """识别标题行，返回 (级别, 标题文本) 或 None。"""
        # Markdown # 标题
        m = re.match(r"^(#{1,3})\s+(.+)$", line)
        if m:
            return len(m.group(1)), _strip_md_inline(m.group(2))
        # 【标题】 格式（常见于 LLM 输出的大标题）
        m = re.match(r"^【(.+?)】(.*)$", line)
        if m:
            extra = m.group(2).strip()
            title = m.group(1) + ("  " + extra if extra else "")
            return 1, _strip_md_inline(title)
        # 整行加粗且以序号开头的小节标题，如 **一、标题**
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

    for line in content.split("\n"):
        stripped = line.strip()

        # 空行
        if not stripped:
            pdf.ln(4)
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

        # 普通段落：去除行内 Markdown 标记
        text = _strip_md_inline(stripped)
        pdf.set_font("Chinese", "", 12)
        pdf.multi_cell(effective_w, 7, text)

    filepath = os.path.join(_output_dir(), f"{_safe_filename(topic)}.pdf")
    pdf.output(filepath)
    return filepath


def save_as_markdown(content: str, topic: str) -> str:
    """将文章内容保存为 Markdown 文件，返回文件路径。"""
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


if __name__ == "__main__":
    selected_llm = choose_llm()

    topic = input("请输入文章主题: ").strip()
    if not topic:
        topic = "人工智能在日常生活中的应用"
        print(f"未输入主题，使用默认主题: {topic}")

    crew = create_crew(topic, selected_llm)
    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("最终文章")
    print("=" * 60)
    print(result)

    try:
        article = str(result)
        md_path = save_as_markdown(article, topic)
        print(f"Markdown 已保存至: {md_path}")
    except Exception as e:
        print(f"\nMarkdown文件保存失败: {e}")

    try:
        article = str(result)
        pdf_path = save_as_pdf(article, topic)
        print(f"\nPDF 已保存至: {pdf_path}")
    except Exception as e:
        print(f"\nPDF文件保存失败: {e}")


