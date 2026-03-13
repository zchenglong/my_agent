# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于 CrewAI 框架的内容创作智能体团队。四个 Agent（研究员 → 作家 → 编辑 → 插画师）以顺序流水线方式协作，将一个主题转化为一篇图文混排的完整文章。当前使用阿里云 DashScope（通义千问）作为 LLM 后端，通义万相作为图片生成后端，同时支持网络搜图（Bing / Pexels）。

## Commands

```bash
# 安装依赖
uv pip install "crewai[tools]" python-dotenv

# 运行（交互式输入主题）
python main.py
```

运行前需在 `.env` 中配置：
- `DASHSCOPE_API_KEY` — 阿里云 DashScope API Key（必需）
- `DASHSCOPE_API_BASE_URL` — API 地址（默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`）
- `PEXELS_API_KEY` — Pexels 图片搜索 API Key（可选，网络搜图优先使用）
- `LOG_LEVEL` — 日志级别（可选，默认 `INFO`，可设为 `DEBUG` 查看详细日志）

## Architecture

`main.py` 是唯一的源文件，包含所有逻辑：

- **`QwenLLM`**: 继承 `BaseLLM` 的自定义千问 LLM 实现，通过 OpenAI SDK 调用 DashScope 兼容接口，实现了 `call()` 抽象方法
- **`default_llm`**: CrewAI 内置 `LLM` 工厂类实例（走 LiteLLM / 原生 provider 路由）
- **`qwen_llm`**: 自定义 `QwenLLM` 实例
- **`choose_llm()`**: 交互式选择使用哪种 LLM 实现
- **`choose_image_source()`**: 交互式选择是否生成配图及图片来源（默认不生成）
- **`build_agents(llm)`**: 构建四个 Agent（researcher / writer / editor / illustrator）
- **`build_tasks(topic, ...)`**: 根据主题构建四个 Task，形成顺序依赖链；插画师挑选 2-3 个最佳配图位置；作家字数范围 1000-5000
- **`create_crew(topic, llm)`**: 组装 Crew，使用 `Process.sequential` 确保任务按序执行
- **`generate_image(prompt, index)`**: 调用 DashScope 通义万相 API 生成配图
- **`_translate_to_keywords(description)`**: 用 Qwen 将中文描述翻译为英文搜索关键词
- **`search_image(description, index)`**: 网络搜图（Pexels API 优先，Bing 抓取兜底）
- **`fetch_image(prompt, index, mode)`**: 根据用户选择的模式调度 AI 生成或网络搜图
- **`save_as_pdf(content, topic, images)`**: 将文章转为 PDF（支持图文混排、列表识别、段间距、副标题样式）
- **`save_as_markdown(content, topic, images)`**: 将文章保存为 Markdown 文件（有图片时转为 `![](path)` 语法，无图片时清除标记）

数据流: 选择 LLM → 输入主题 → research_task → write_task → edit_task → illustrate_task → 选择是否生成配图 → 输出 PDF + Markdown

## Key Conventions

- 所有 Agent 共享同一个 LLM 实例，通过 `llm=` 参数注入
- Agent 的 `allow_delegation=False` 防止任务在 Agent 间互相委派
- 所有 Agent prompt 使用中文，生成中文内容
- 环境变量通过 `python-dotenv` 从 `.env` 加载
- `[IMG: ...]` 和 `【IMG: ...】` 两种标记格式均支持
- 第三方库日志（fontTools、httpx 等）默认压到 WARNING 级别
