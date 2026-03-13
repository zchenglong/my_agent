# My Agent - 内容创作智能体团队

基于 [CrewAI](https://github.com/crewAIInc/crewAI) 框架的内容创作智能体团队。四个 Agent（研究员 → 作家 → 编辑 → 插画师）以顺序流水线方式协作，将一个主题转化为一篇图文混排的完整文章。

## 工作流程

```
输入主题 → 研究员（素材搜集） → 作家（文章撰写） → 编辑（审校定稿） → 插画师（配图策划） → 生成配图（可选） → 输出 PDF + Markdown
```

- **资深内容研究员** — 围绕主题搜集全面、准确、有深度的素材和观点
- **资深内容作家** — 根据研究素材撰写 1000-5000 字的高质量文章
- **资深内容编辑** — 审校文章质量，确保内容准确、结构合理、语言精炼
- **资深插画策划师** — 为文章挑选 2-3 个最佳配图位置，设计图片描述

## 环境要求

- Python >= 3.10, < 3.14

## 安装

```bash
# 创建虚拟环境
uv venv

# 安装依赖
uv pip install "crewai[tools]" python-dotenv
```

## 配置

在项目根目录创建 `.env` 文件：

```env
# 必需
DASHSCOPE_API_KEY=sk-xxx
DASHSCOPE_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 可选 - Pexels 图片搜索（网络搜图时优先使用，无此项则用 Bing 搜索兜底）
PEXELS_API_KEY=xxx

# 可选 - 日志级别（默认 INFO，可设为 DEBUG 查看详细日志）
LOG_LEVEL=INFO
```

## 运行

```bash
python main.py
```

运行后会依次提示：

1. **选择 LLM 实现方式**（CrewAI 内置 LLM 或自定义 QwenLLM）
2. **输入文章主题**（留空使用默认主题「人工智能在日常生活中的应用」）
3. **选择是否生成配图**（默认不生成）

### LLM 实现方式

| 选项 | 说明 |
|------|------|
| CrewAI 内置 LLM | 使用 CrewAI 的 `LLM` 工厂类，底层自动路由到对应 provider |
| 自定义 QwenLLM | 继承 `BaseLLM`，通过 OpenAI SDK 直接调用 DashScope 兼容接口 |

### 配图选项

| 选项 | 说明 |
|------|------|
| 不生成配图（默认） | 跳过图片生成，PDF/Markdown 中不包含图片 |
| AI 生成优先 | 通义万相生成，失败后回退到网络搜图 |
| 网络搜图优先 | Pexels/Bing 搜图，失败后回退到 AI 生成 |
| 仅 AI 生成 | 只使用通义万相 |
| 仅网络搜图 | 只使用 Pexels/Bing 搜索 |

### 输出格式

- **PDF** — 支持中文字体、标题层级、列表识别、段间距、图文混排
- **Markdown** — 有图片时自动转为 `![描述](路径)` 语法，无图片时清除标记

输出文件保存在 `output/` 目录。

## 项目结构

```
.
├── main.py          # 主程序，包含所有 Agent、Task、Crew 定义
├── pyproject.toml   # 项目配置与依赖声明
├── CLAUDE.md        # Claude Code 引导文件
├── .env             # 环境变量配置（不纳入版本控制）
├── output/          # 生成的文章和图片
└── README.md
```

## License

MIT
