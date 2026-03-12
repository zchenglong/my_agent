# My Agent - 内容创作智能体团队

基于 [CrewAI](https://github.com/crewAIInc/crewAI) 框架的内容创作智能体团队。三个 Agent（研究员 → 作家 → 编辑）以顺序流水线方式协作，将一个主题转化为一篇完整文章。

## 工作流程

```
输入主题 → 研究员（素材搜集） → 作家（文章撰写） → 编辑（审校定稿） → 最终文章
```

- **资深内容研究员** — 围绕主题搜集全面、准确、有深度的素材和观点
- **资深内容作家** — 根据研究素材撰写结构清晰、引人入胜的高质量文章
- **资深内容编辑** — 审校文章质量，确保内容准确、结构合理、语言精炼

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

在项目根目录创建 `.env` 文件，配置对应模型的 API Key：

```env
DASHSCOPE_API_KEY=sk-xxx
DASHSCOPE_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 运行

```bash
python main.py
```

运行后会依次提示：
1. 选择 LLM 实现方式（CrewAI 内置 LLM 或自定义 QwenLLM）
2. 输入文章主题（留空使用默认主题「人工智能在日常生活中的应用」）

### LLM 实现方式

| 选项 | 说明 |
|------|------|
| CrewAI 内置 LLM | 使用 CrewAI 的 `LLM` 工厂类，底层自动路由到对应 provider |
| 自定义 QwenLLM | 继承 `BaseLLM`，通过 OpenAI SDK 直接调用 DashScope 兼容接口 |

## 项目结构

```
.
├── main.py          # 主程序，包含所有 Agent、Task、Crew 定义
├── pyproject.toml   # 项目配置与依赖声明
├── CLAUDE.md        # Claude Code 引导文件
├── .env             # 环境变量配置（不纳入版本控制）
└── README.md
```

## License

MIT
