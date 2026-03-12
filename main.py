"""
Content Creation Crew - 内容创作智能体团队

使用 CrewAI 框架构建的内容创作流水线：
研究员 → 作家 → 编辑

支持多种 LLM：OpenAI / Anthropic Claude / DeepSeek / Ollama 本地模型
"""

import logging
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

load_dotenv(override=True)

# ── LLM ──────────────────────────────────────────────

_api_key = os.getenv("DASHSCOPE_API_KEY")
_base_url = os.getenv("DASHSCOPE_API_BASE_URL")

if not _api_key:
    raise SystemExit(
        "错误: 未设置 DASHSCOPE_API_KEY。\n"
        "请在 .env 文件中添加，或通过环境变量导出:\n"
        "  export DASHSCOPE_API_KEY=sk-xxx\n"
        "  export DASHSCOPE_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

llm = LLM(
    model="qwen-plus",
    api_key=_api_key,
    base_url=_base_url,
)

# ── Agents ──────────────────────────────────────────────

def build_agents(llm: LLM) -> tuple[Agent, Agent, Agent]:
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

def create_crew(topic: str, llm: LLM) -> Crew:
    """创建内容创作 Crew。"""
    researcher, writer, editor = build_agents(llm)
    return Crew(
        agents=[researcher, writer, editor],
        tasks=build_tasks(topic, researcher, writer, editor),
        process=Process.sequential,
        verbose=True,
    )


# ── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    topic = input("请输入文章主题: ").strip()
    if not topic:
        topic = "人工智能在日常生活中的应用"
        print(f"未输入主题，使用默认主题: {topic}")
    # topic = "人工智能在日常生活中的应用"
    crew = create_crew(topic, llm)
    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("最终文章")
    print("=" * 60)
    print(result)
