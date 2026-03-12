# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于 CrewAI 框架的内容创作智能体团队。三个 Agent（研究员 → 作家 → 编辑）以顺序流水线方式协作，将一个主题转化为一篇完整文章。支持 OpenAI / Anthropic Claude / DeepSeek / Ollama 本地模型。

## Commands

```bash
# 安装依赖
uv pip install "crewai[tools]" python-dotenv

# 运行（交互式选择模型和主题）
.venv/bin/python main.py
```

运行前需在 `.env` 中配置对应模型的 API Key（如 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`DEEPSEEK_API_KEY`）。

## Architecture

`main.py` 是唯一的源文件，包含所有逻辑：

- **`AVAILABLE_MODELS`**: 模型注册表，定义所有可选的 LLM（model_id → 显示名称）
- **`choose_llm()`**: 交互式模型选择，根据 provider 自动处理特殊参数（Anthropic 的 max_tokens、Ollama 的 base_url）
- **`build_agents(llm)`**: 接收 LLM 实例，构建三个 Agent（researcher / writer / editor）
- **`build_tasks(topic, ...)`**: 根据主题动态构建三个 Task，形成顺序依赖链
- **`create_crew(topic, llm)`**: 组装 Crew，使用 `Process.sequential` 确保任务按序执行

数据流: 选择模型 → 输入主题 → research_task → write_task → edit_task → 最终文章输出

## Key Conventions

- 所有 Agent 共享同一个 LLM 实例，通过 `llm=` 参数注入
- Agent 的 `allow_delegation=False` 防止任务在 Agent 间互相委派
- 所有 Agent prompt 使用中文，生成中文内容
- 环境变量通过 `python-dotenv` 从 `.env` 加载
- 模型标识符需带 provider 前缀（如 `openai/gpt-4o`、`anthropic/claude-sonnet-4-20250514`）
