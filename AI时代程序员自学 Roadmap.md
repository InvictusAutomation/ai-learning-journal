# AI 时代程序员自学 Roadmap

> 面向 AI 时代的技术专家、工程师、架构师的自学成才路线图

---

## 核心理念转变

| 传统编程 | AI 时代编程 |
|---------|-------------|
| 手写代码 | 生成 + 修改 + 审查 |
| 调试代码 | 调优上下文 |
| 单元测试 | 自然语言验证 |
| 算法优先 | 提示词优先 |
| 工具调用 | Agent 编排 |

---

## 学习路径总览

```
AI 时代程序员成长路径
│
├── 阶段一：AI 助手入门 (1-2周)
│   │
│   ├── 1.1 掌握一个 AI coding agent
│   ├── 1.2 理解 prompts 机制
│   └── 1.3 建立 AI 协作工作流
│
├── 阶段二：AI 编程基础 (2-4周)
│   │
│   ├── 2.1 提示词工程
│   ├── 2.2 RAG 基础
│   └── 2.3 工具调用开发
│
├── 阶段三：AI Agent 开发 (4-8周)
│   │
│   ├── 3.1 Agent 框架
│   ├── 3.2 多 Agent 系统
│   └── 3.3 自主决策循环
│
├── 阶段四：AI 工程实践 (8-12周)
│   │
│   ├── 4.1 生产级 RAG 系统
│   ├── 4.2 工具生态集成
│   └── 4.3 评估与优化
│
└── 阶段五：AI 架构专家 (持续)
    │
    ├── 5.1 大型 Agent 系统设计
    ├── 5.2 AI 基础设施
    └── 5.3 前沿研究与创新
```

---

## 阶段一：AI 助手入门

### 目标：熟练使用 AI coding tool

**推荐工具**：

| 工具 | 场景 | 特点 |
|------|------|------|
| **Claude Code** | 专业开发 | 代码质量高 |
| **OpenCode** | 全栈开发 | 中文友好 |
| **Cursor** | IDE 集成 | 实时补全 |
| **Windsurf** | 创新体验 | Agent 模式 |

### 核心技能

```
□ 安装和配置 AI coding tool
□ 理解 agent 模式 (act / recommend / auto)
□ 学会编写有效的 prompts
□ 理解系统 prompt 和约束
□ 审查 AI 生成的代码
□ 使用 agent 自动功能
```

### 练习项目

1. 用 AI agent 完成一个小功能
2. 让 agent 修复一个已知 bug
3. 使用 agent 进行代码重构

### 参考资源

- [OpenClaw 文档](https://docs.openclaw.ai)
- [Claude Code 官方](https://docs.anthropic.com/en/docs/claude-code/overview)
- [OpenCode 文档](https://opencode.ai/docs)

---

## 阶段二：AI 编程基础

### 2.1 提示词工程

**CO-STAR 框架**：

| 元素 | 问题 | 示例 |
|------|------|------|
| **C**ontext | 什么场景？ | "你是 10 年经验架构师" |
| **O**bjective | 目标？ | "审查微服务架构" |
| **S**tyle | 风格？ | "技术博客风格" |
| **T**one | 语气？ | "专业但易懂" |
| **A**udience | 受众？ | "初中级工程师" |
| **R**esponse | 格式？ | "Markdown 表格" |

**ReAct 模式**：

```
Thought: 我需要先理解问题
Action: search(相关资料)
Observation: 找到 3 个相关文档
Thought: 根据文档分析
Action: summarize(observation)
Final Answer: ...
```

### 2.2 RAG 基础

```
┌──────────────┐    ┌──────────────┐
│   用户问题   │───▶│   向量化    │
└──────────────┘    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  向量检索    │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  LLM 生成   │
                    └─────���────────┘
```

**向量数据库选择**：

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| Pinecone | 全托管 | 生产环境 |
| Milvus | 开源 | 大规模部署 |
| Qdrant | 轻量 | 个人项目 |
| Chroma | 简单 | 快速原型 |

### 2.3 工具调用开发

**OpenAI Function Calling**：

```python
# 定义工具
functions = [
    {
        "name": "get_weather",
        "description": "获取天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["location"]
        }
    }
]

# 调用
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京天气如何？"}],
    functions=functions
)
```

---

## 阶段三：AI Agent 开发

### 3.1 Agent 框架对比

| 框架 | 语言 | 特点 | 适合 |
|------|------|------|------|
| **LangChain** | Python | 全功能 | 生产 |
| **LangGraph** | Python | 可视化 | 复杂工作流 |
| **LlamaIndex** | Python | 索引优化 | RAG |
| **AutoGen** | Python | 多 Agent | 协作 |
| **CrewAI** | Python | 角色扮演 | 团队 |
| **OpenClaw** | TypeScript | 多平台 | 个人助手 |

### 3.2 多 Agent 系统架构

```
┌─────────────────────────────────────────────┐
│              Gateway / Router              │
│         (根据意图分发到合适的 Agent)       │
└─────────────────────┬───────────────────────┘
                      │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│Research │    │ Coder   │    │ Writer  │
│ Agent   │    │ Agent  │    │ Agent  │
└────┬────┘    └────┬────┘    └────┬────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
              ┌──────┴──────┐
              │ Orchestrator│
              │    Agent    │
              └─────────────┘
```

### 3.3 自主决策循环

```python
# Agent 自主循环
class AgentLoop:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.max_iterations = 10
        
    def run(self, task):
        history = []
        
        for i in range(self.max_iterations):
            # 1. 思考
            thought = self.llm.think(task, history)
            
            # 2. 行动
            if thought.is_finished():
                return thought.result
                
            action = self.llm.decide_action(thought, self.tools)
            
            # 3. 执行
            result = action.execute()
            
            # 4. 观察
            history.append({
                "thought": thought,
                "action": action,
                "result": result
            })
            
        return "达到最大迭代次数"
```

---

## 阶段四：AI 工程实践

### 4.1 生产级 RAG 系统

**高级 RAG 技术** (来自 [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques))：

| 技术 | 描述 | 难度 |
|------|------|------|
| **Agentic RAG** | Agent 驱动的自主 RAG | ⭐⭐⭐⭐⭐ |
| **Self-RAG** | 自反思检索增强 | ⭐⭐⭐⭐ |
| **Graph RAG** | 知识图谱增强 | ⭐⭐⭐⭐⭐ |
| **HyDE** | 假设文档嵌入 | ⭐⭐⭐ |
| **Reranking** | 两阶段重排 | ⭐⭐⭐ |

### 4.2 工具生态集成

```
┌─────────────────────────────────────────────────────┐
│                  AI Agent                          │
├─────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│
│  │  exec  │  │  file   │  │  http   │  │ browser││
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘│
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│
│  │   sql   │  │ vector  │  │ message │  │  nodes ││
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘│
└─────────────────────────────────────────────────────┘
```

### 4.3 评估与优化

**RAG 评估指标**：

| 指标 | 描述 | 工具 |
|------|------|------|
| **Faithfulness** | 答案来自检索内容 | DeepEval |
| **Answer Relevance** | 答案与问题相关 | DeepEval |
| **Context Precision** | 检索精准度 | DeepEval |
| **Context Recall** | 检索召回率 | DeepEval |

---

## 阶段五：AI 架构专家

### 5.1 大型系统设计

**架构考量**：

```
┌─────────────────────────────────────────────────────┐
│              AI Platform Architecture             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │            API Gateway                       │   │
│  │    (认证、限流、路由、日志)                   │   │
│  └─────────────────────┬───────────────────────┘   │
│                        │                            │
│  ┌─────────────────────┴───────────────────────┐   │
│  │            Agent Orchestrator              │   │
│  │       (任务分解、调度、协调)                │   │
│  └─────────────────────┬───────────────────────┘   │
│                        │                            │
│  ┌────────┬───────────┼───────────┬────────┐       │
│  │        │           │           │        │       │
│  ▼        ▼           ▼           ▼        ▼       │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                    │
│ │Tool│ │Tool│ │Tool│ │Tool│ │Tool│                    │
│ │Agent│Agent│Agent│Agent│Agent│                    │
│ └────┘ └────┘ └────┘ └────┘ └────┘                    │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │         Data Layer                         │   │
│  │   (向量库、知识图、日志、监控)             │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 5.2 AI 基础设施

| 组件 | 技术选型 | 备注 |
|------|---------|------|
| **模型服务** | vLLM, TGI | 高吞吐量推理 |
| **向量检索** | Milvus, Pinecone | 规模化检索 |
| **知识图谱** | Neo4j | 关系推理 |
| **监控** | LangSmith, Phoenix | 可观测性 |
| **部署** | Kubernetes, Ray Serve | 弹性伸缩 |

### 5.3 前沿研究

**保持跟进**：

- [OpenClaw](https://github.com/openclaw/openclaw) - AI 助手框架
- [LangChain](https://github.com/langchain-ai/langchain) - Agent 框架
- [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) - RAG 技术
- [Superpowers](https://github.com/obra/superpowers) - Agent 工作流

---

## 技能清单

### 基础技能 (必备)

- [ ] 熟练使用至少一个 AI coding tool
- [ ] 理解提示词工程原理
- [ ] 掌握基础 RAG 实现
- [ ] 能够调用 LLM API

### 中级技能 (推荐)

- [ ] 使用 Agent 框架构建应用
- [ ] 实现多 Agent 协作
- [ ] 构建生产级 RAG 系统
- [ ] 掌握评估方法

### 高级技能 (进阶)

- [ ] 设计大型 AI 系统架构
- [ ] 优化模型推理性能
- [ ] 前沿技术研究能力
- [ ] 团队 AI 赋能

---

## 参考资源

### GitHub 优秀 Roadmap

| 仓库 | Stars | 主题 |
|------|-------|------|
| [AMAI-GmbH/AI-Expert-Roadmap](https://github.com/AMAI-GmbH/AI-Expert-Roadmap) | 30K | AI 专家路线 |
| [krishnaik06/Roadmap-To-Learn-Generative-AI-In-2025](https://github.com/krishnaik06/Roadmap-To-Learn-Generative-AI-In-2025) | 5K | 生成式 AI |
| [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) | 101K | LLM 应用集合 |
| [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | 26K | RAG 技术 |

### 本仓库相关文档

- [AI编程时代核心技术能力](AI编程时代核心技术能力.md)
- [AI编程时代核心技术能力_深度版](AI编程时代核心技术能力_深度版.md)

---

*本文档由 OpenClaw AI 学习系统自动生成并维护*
*持续更新中...*