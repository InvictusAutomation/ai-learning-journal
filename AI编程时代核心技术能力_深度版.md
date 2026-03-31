# AI 编程时代核心技术能力 (深度扩展版)

> 基于 GitHub 前沿仓库和论文研究的深度技术文档

---

## 一、RAG 高级技术 (来自 NirDiamant/RAG_Techniques)

### 1.1 高级检索技术分类

| # | 技术 | 描述 | 实现难度 |
|---|------|------|--------|
| 1 | **Agentic RAG** | Agent 驱动的自主 RAG，可自我反思和迭代 | ⭐⭐⭐⭐⭐ |
| 2 | **Self-RAG** | 自反思检索增强生成，按需检索 | ⭐⭐⭐⭐ |
| 3 | **Corrective RAG (CRAG)** | 自动检测和纠正检索错误 | ⭐⭐⭐⭐ |
| 4 | **Graph RAG** | 知识图谱增强的 RAG | ⭐⭐⭐⭐⭐ |
| 5 | **Microsoft GraphRAG** | 企业级知识图谱 RAG | ⭐⭐⭐⭐⭐ |
| 6 | **RAPTOR** | 递归抽象树检索 | ⭐⭐⭐⭐⭐ |
| 7 | **HyDE** | 假设文档嵌入 | ⭐⭐⭐ |
| 8 | **HyPE** | 假设提示嵌入 | ⭐⭐⭐ |
| 9 | **Fusion Retrieval** | 多检索器融合 | ⭐⭐⭐⭐ |
| 10 | **Reranking** | 两阶段检索重排 | ⭐⭐⭐ |
| 11 | **Adaptive Retrieval** | 自适应检索策略 | ⭐⭐⭐⭐ |
| 12 | **Iterative Retrieval** | 迭代检索反馈 | ⭐⭐⭐⭐ |

### 1.2 Agentic RAG 架构

```
┌─────────────────────────────────────────────────────────────┐
│                  Query Input                           │
└─────────────────────┬─────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Planner Agent                             │
│  - 分析用户意图                                         │
│  - 确定检索策略                                         │
│  - 规划执行步骤                                         │
└─────────────────────┬─────────────────────────────────┘
                      │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Vector Search   │     │ Knowledge Graph│
│ (向量化检索)    │     │ (图谱检索)      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Judge Agent (判断器)                           │
│  - 评估检索质量                                          │
│  - 决定是否需要二次检索                                    │
│  - 验证答案相关性                                         │
└─────────────────────┬─────────────────────────────────────┘
                      │
             ┌────────┴────────┐
             ▼                 ▼
┌──────────────┐    ┌──────────────┐
│  答案可用    │    │  需要重检索   │
│  (Generate) │    │   (Loop)     │
└──────────────┘    └──────────────┘
```

### 1.3 核心 RAG 技术实现

#### 1.3.1 HyDE (Hypothetical Document Embedding)

```python
# HyDE 核心思想：让 LLM 生成一个"假设文档"，用这个文档去检索
def hyde_retrieve(query, vector_store, llm):
    # Step 1: 生成假设文档
    hypothetical_doc = llm.generate(f"""
    生成一个理想回答以下问题的文档：
    
    问题: {query}
    
    假设这是一个知识库文档，直接给出答案：""")
    
    # Step 2: 向量化假设文档
    query_embedding = embed(hypothetical_doc)
    
    # Step 3: 检索相似文档
    results = vector_store.similarity_search(query_embedding)
    
    return results
```

**优势**：
- 解决查询与文档语义空间不匹配问题
- 查询改写 + 语义检索的结合
- 在专业领域效果显著

#### 1.3.2 Chunking 技术对比

| 技术 | 原理 | 适用场景 |
|------|------|----------|
| **Fixed-size** | 固定 token 数分割 | 通用场景 |
| **Sentence** | 按句子分割 | 精确问答 |
| **Semantic** | 语义边界分割 | 复杂文档 |
| **Proposition** | 命题分割 | 知识抽取 |
| **Contextual** | 上下文感知分割 | 大文档 |

```python
# Proposition Chunking 示例
def proposition_chunking(text, llm):
    """将文本分解为独立的命题"""
    propositions = llm.generate(f"""
    将以下文本分解为独立的原子命题：
    文本: {text}
    
    格式要求：
    - 每个命题应该是一个完整的陈述
    - 不包含代词引用
    - 保持原始语义
    """)
    return propositions.split("\n")
```

### 1.4 RAG 评估指标

| 指标 | 描述 | 工具 |
|------|------|------|
| **Faithfulness** | 答案是否来自检索内容 | DeepEval |
| **Answer Relevance** | 答案与问题的相关度 | DeepEval |
| **Context Precision** | 相关文档的精准度 | DeepEval |
| **Context Recall** | 检索召回率 | DeepEval |
| **GroUSE** | 生成质量评分 | GroUSE |

---

## 二、AI Agent 核心架构 (来自 OpenCode, Superpowers)

### 2.1 OpenCode 架构分析

**核心特性**：
- 多文件编辑能力
- 安全优先的沙盒执行
- 双向状态同步
- Token 效率优化

```yaml
# OpenCode 工作流
workflow:
  - name: Understand
    prompt: "理解需求和现有代码结构"
    
  - name: Plan
    prompt: "制定修改计划，识别需改动的文件"
    
  - name: Execute  
    prompt: "执行代码修改"
    tools:
      - file_write
      - bash_execute
      
  - name: Verify
    prompt: "验证修改正确性"
    tools:
      - test_run
      - lint_check
```

### 2.2 Superpowers Agent Skill 框架

```python
# Superpowers 定义
skill = {
    "name": "代码审查",
    "description": "专业的代码审查技能",
    "instructions": """
    你是一个高级代码审查专家。
    
    审查维度：
    1. 代码质量 - 可读性、可维护性
    2. 性能 - 时间/空间复杂度
    3. 安全 - 漏洞检查
    4. 最佳实践 - 遵循项目规范
    
    输出格式：
    - 问题列表
    - 严重程度 (critical/high/medium/low)
    - 修复建议
    """,
    "tools": ["file_read", "search"],
    "examples": [
        {
            "input": "审查这个函数",
            "output": "发现3个问题..."
        }
    ]
}
```

### 2.3 多 Agent 通信协议

#### A2A Protocol (Agent to Agent)

```json
{
  "jsonrpc": "2.0",
  "id": "msg-001",
  "method": "tasks/send",
  "params": {
    "id": "task-123",
    "message": {
      "role": "user",
      "parts": [{
        "type": "text",
        "text": "分析这个代码的性能问题"
      }]
    },
    "history": [
      {"role": "assistant", "parts": [...]}
    ]
  }
}
```

#### 响应格式

```json
{
  "jsonrpc": "2.0", 
  "id": "msg-001",
  "result": {
    "id": "task-123",
    "status": "completed",
    "message": {
      "role": "assistant", 
      "parts": [{
        "type": "text", 
        "content": "分析结果..."
      }]
    }
  }
}
```

---

## 三、Browser Use 与网页自动化

### 3.1 Browser Use 架构

```python
# Browser Use 核心组件
class BrowserAgent:
    def __init__(self):
        self.browser_config = {...}
        self.stealth_mode = True
        self.max_steps = 100
        
    async def navigate_and_extract(self, url, goal):
        """导航到页面并提取信息"""
        page = await self.browser.new_page()
        
        await page.goto(url, wait_until="networkidle")
        
        # 执行目标动作
        await self.execute_goal(page, goal)
        
        return await self.extract_results(page)
```

### 3.2 常见用例

| 用例 | 描述 |
|------|------|
| 表单填写 | 自动填写和提交表单 |
| 数据抓取 | 从网页提取结构化数据 |
| 自动化测试 | UI 测试自动化 |
| 社交媒体管理 | 自动发帖、互动 |

---

## 四、提示词工程深度技术

### 4.1 CO-STAR 框架详解

| 元素 | 问题 | 示例 |
|------|------|------|
| **C** Context | 什么场景？ | "你是一个有10年经验的高级软件架构师" |
| **O** Objective | 目标是什么？ | "审查以下微服务架构设计" |
| **S** Style | 什么风格？ | "技术博客风格，详实但不过于学术" |
| **T** Tone | 什么语气？ | "专业、友善、带有建设性" |
| **A** Audience | 给谁看？ | "初级到中级开发工程师" |
| **R** Response | 什么格式？ | "Markdown 表格，包含评分和建议" |

### 4.2 ReAct (Reasoning + Acting)

```python
# ReAct 实现
def react_agent(question, tools, llm):
    thought = f"思考: {question}"
    
    # 循环执行直到得到答案
    while True:
        # 推理下一步
        action = llm.decide(thought, question, tools)
        
        # 执行动作
        if action.type == "search":
            result = action.execute()
        elif action.type == "finish":
            return action.answer
            
        # 观察结果
        observation = f"观察到: {result}"
        thought = f"{thought}\n{observation}"
```

### 4.3 Chain of Thought 变体

| 技术 | 描述 | 适用场景 |
|------|------|----------|
| **CoT** | 逐步推理 | 数学、逻辑问题 |
| **CoT-SC** | 自一致性 | 需要准确答案 |
| **CoT-Zero** | 无示例推理 | 通用问题 |
| **ToT** | 树状探索 | 复杂决策 |
| **GoT** | 图状推理 | 多角度分析 |
| **XoT** | 外部知识增强 | 知识密集任务 |

---

## 五、上下文工程 (Context Engineering)

### 5.1 Memory 分层架构

```
┌─────────────────────────────────────────────────────┐
│                 Long-term Memory                   │
│  (长期记忆: MEMORY.md, 知识库)                       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│               Session Context                     │
│  (会话上下文: 最近 N 轮对话)                        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                 Working Memory                     │
│  (工作记忆: 当前任务状态)                           │
└─────────────────────────────────────────────────────┘
```

### 5.2 Context 压缩策略

```python
class ContextCompressor:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        
    def compress(self, messages):
        """压缩对话历史"""
        # 1. 重要性评分
        scored = self.score_importance(messages)
        
        # 2. 选择高重要性消息
        selected = self.select_top(scored, self.max_tokens)
        
        # 3. 摘要压缩
        summarized = self.summarize(selected)
        
        return summarized
    
    def score_importance(self, messages):
        """基于多种因子评分"""
        scores = []
        for msg in messages:
            score = 0
            if msg.has_code(): score += 3
            if msg.has_error(): score += 5
            if msg.is_user_final(): score += 4
            if msg.tool_used(): score += 2
            scores.append(score)
        return scores
```

### 5.3 滑动窗口策略

```python
# 分层滑动窗口
def build_context(user_query, session_history):
    # 近几轮完整保留
    recent = session_history[-3:]
    
    # 中间轮次摘要
    middle = summarize(session_history[-10:-3])
    
    # 远轮次只保留意图
    remote = [m.intent for m in session_history[:-10]]
    
    return f"""
{context_for(recent)}
{middle_summary}
{remote_intents}

当前问题: {user_query}
"""
```

---

## 六、Tool Use 与工具生态

### 6.1 工具分类

| 类别 | 工具 | 功能 |
|------|------|------|
| **执行** | exec, process | 运行代码和命令 |
| **文件** | read, write, edit | 文件操作 |
| **网络** | http, web_fetch | API 调用 |
| **浏览器** | browser | 网页自动化 |
| **数据库** | sql, vector_db | 数据查询 |
| **消息** | message | 多平台消息 |
| **设备** | nodes | 设备控制 |

### 6.2 工具选择策略

```python
def select_tools(task, available_tools):
    # 1. 分析任务类型
    task_type = classify(task)
    
    # 2. 匹配工具能力
    matching_tools = match(task_type, available_tools)
    
    # 3. 排序和选择
    ranked = rank_by_confidence(matching_tools)
    
    # 4. 返回 Top-K
    return ranked[:3]
```

### 6.3 工具组合模式

| 模式 | 描述 | 示例 |
|------|------|------|
| **Sequential** | 顺序执行 | 读取 → 处理 → 写入 |
| **Parallel** | 并行执行 | 同时调用多个 API |
| **Conditional** | 条件分支 | 根据结果选择下一个工具 |
| **Retry** | 带重试 | 失败自动重试 |
| **Fallback** | 降级 | 主工具失败用备选 |

---

## 七、模型提供商与选择

### 7.1 主流模型对比

| 提供商 | 模型 | 特点 | 适用场景 |
|--------|------|------|----------|
| **OpenAI** | o1, o3-mini | 推理强 | 复杂推理 |
| **Anthropic** | Claude 4 | 长上下文 | 代码/文档 |
| **Google** | Gemini 2.0 | 多模态 | 综合性 |
| **DeepSeek** | V3 | 开源性价比高 | 生产部署 |
| **MiniMax** | M2.1 | 中文优化 | 中文场景 |

### 7.2 模型选择决策树

```
问题复杂度
    │
    ├─ 简单 (单轮问答) → 小模型 (快+便宜)
    │      │
    │      └─ Claude 3.5 Haiku / GPT-4o Mini
    │
    ├─ 中等 (多轮对话) → 中等模型
    │      │
    │      └─ Claude 3.5 Sonnet / GPT-4o
    │
    └─ 复杂 (推理/代码) → 大模型
           │
           └─ Claude 4 Opus / o1 / o3-mini
```

---

## 八、可观测性与调试

### 8.1 Agent 调试指标

| 指标 | 描述 |
|------|------|
| **Token 消耗** | 每个环节消耗 |
| **执行时长** | 各步骤耗时 |
| **工具调用** | 工具使用次数/成功率 |
| **错误率** | 失败模式分析 |
| **重试次数** | 自动恢复情况 |

### 8.2 日志结构

```python
# 结构化日志
log = {
    "timestamp": "2026-04-01T10:00:00Z",
    "session_id": "sess-001",
    "turn": 5,
    "step": {
        "thought": "需要读取文件",
        "action": "file_read",
        "params": {"path": "/src/main.py"},
        "result": "成功读取 200 行"
    },
    "tokens": {
        "input": 5000,
        "output": 2000
    },
    "latency_ms": 1500
}
```

---

## 九、参考资料与深入学习

### 9.1 GitHub 前沿仓库

| 仓库 | Stars | 主题 |
|------|------|------|
| [openclaw/openclaw](https://github.com/openclaw/openclaw) | 342K | AI 助手框架 |
| [x1xhlol/system-prompts-and-models-of-ai-tools](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools) | 133K | 系统提示集合 |
| [anomalyco/opencode](https://github.com/anomalyco/opencode) | 133K | AI 编程 Agent |
| [obra/superpowers](https://github.com/obra/superpowers) | 124K | Agent Skill 框架 |
| [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | 26K | RAG 技术集合 |
| [browser-use/browser-use](https://github.com/browser-use/browser-use) | 85K | 浏览器自动化 |
| [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) | 101K | LLM 应用集合 |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | 62K | AI 研究 Agent |

### 9.2 论文推荐

| 论文 | 主题 |
|------|------|
| **Self-RAG** | 按需检索增强 |
| **RAPTOR** | 递归树检索 |
| **Chain of Thought** | 思维链推理 |
| **ReAct** | 推理+行动 |
| **Tree of Thoughts** | 树状思考 |
| **Graph RAG** | 知识图谱 RAG |

### 9.3 在线资源

- [LangChain Documentation](https://python.langchain.com)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/preset-chat-constructing-chats-with-tool-use)

---

*本文档持续更新中 - 来源于 GitHub 前沿研究和实践*