# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于 CrewAI 框架的内容创作智能体团队。三个 Agent（研究员 → 作家 → 编辑）以顺序流水线方式协作，将一个主题转化为一篇完整文章。当前使用阿里云 DashScope（通义千问）作为 LLM 后端。

## Commands

```bash
# 安装依赖
uv pip install "crewai[tools]" python-dotenv

# 运行（交互式输入主题）
python main.py
```

运行前需在 `.env` 中配置：
- `DASHSCOPE_API_KEY` — 阿里云 DashScope API Key
- `DASHSCOPE_API_BASE_URL` — API 地址（默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`）

## Architecture

`main.py` 是唯一的源文件，包含所有逻辑：

- **`llm`**: 全局 LLM 实例，使用 DashScope qwen-plus 模型
- **`build_agents(llm)`**: 构建三个 Agent（researcher / writer / editor）
- **`build_tasks(topic, ...)`**: 根据主题构建三个 Task，形成顺序依赖链
- **`create_crew(topic, llm)`**: 组装 Crew，使用 `Process.sequential` 确保任务按序执行

数据流: 输入主题 → research_task → write_task → edit_task → 最终文章输出

## Key Conventions

- 所有 Agent 共享同一个 LLM 实例，通过 `llm=` 参数注入
- Agent 的 `allow_delegation=False` 防止任务在 Agent 间互相委派
- 所有 Agent prompt 使用中文，生成中文内容
- 环境变量通过 `python-dotenv` 从 `.env` 加载
