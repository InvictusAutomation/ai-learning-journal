# 第一部分：提示词工程 (Prompt Engineering)

> 提示词工程是 AI 时代的核心编程技能 - 决定了 AI 输出的质量和准确性

---

## 1.1 提示词工程基础

### 什么是提示词工程？

提示词工程 (Prompt Engineering) 是设计与优化 AI 提示词的艺术和科学，用于从语言模型获取高质量输出的实践。

### 核心原则

| 原则 | 描述 | 示例 |
|------|------|------|
| **清晰具体** | 明确说明需求 | "写一个函数" vs "写一个 Python 函数，输入列表返回平均值" |
| **提供上下文** | 给足背景信息 | "你是一个有10年经验的..." |
| **结构化输出** | 指定输出格式 | "用 JSON 格式返回" |
| **分步思考** | 引导推理过程 | "一步一步思考" |
| **few-shot** | 提供示例 | 给3个输入输出对 |

---

## 1.2 CO-STAR 框架详解

### 框架要素

```
┌─────────────────────────────────────────────────────────────┐
│                  CO-STAR 框架                          │
├─────────────────────────────────────────────────────────────┤
│                                                     │
│  C - Context (上下文)                                 │
│     └── 我是谁？做什么的？有什么背景？               │
│                                                     │
│  O - Objective (目标)                                 │
│     └── 我要做什么？完成任务是什么？                  │
│                                                     │
│  S - Style (风格)                                     │
│     └── 用什么风格？技术博客？学术论文？           │
│                                                     │
│  T - Tone (语气)                                      │
│     └── 什么语气？正式？轻松？建议？                │
│                                                     │
│  A - Audience (受众)                                  │
│     └── 谁在看？专家？小白？                        │
│                                                     │
│  R - Response (响应)                                 │
│     └── 怎么返回？Markdown？表格？                │
│                                                     │
└─────────────────────────────────────────────────────────────┘
```

### 应用示例

**原始 Prompt**：
```
写一个排序算法
```

**CO-STAR 优化后**：
```
你是一个有15年经验的高级软件工程师 (C)
帮我写一个高效的排序算法，用于处理大规模数据集 (O)
用技术博客风格，包含代码和注释 (S)
专业且易于理解 (T)
初中级开发工程师 (A)
用 Python 代码 + Markdown 说明格式 (R)
```

---

## 1.3 提示词模式对比

### 1.3.1 Zero-Shot Prompting

```python
# 不给任何示例，直接让模型回答
prompt = "把这句话翻译成英文：我喜欢编程"
response = llm.generate(prompt)
```

**适用场景**：简单任务、日常对话

### 1.3.2 Few-Shot Prompting

```python
# 提供示例
prompt = """
例子1:
输入：今天天气怎么样？
输出：今天天气晴朗，气温20-28度

例子2:
输入：明天会下雨吗？
输出：明天有小雨，气温18-25度

现在请回答：
输入：���天适合出门吗？
"""
response = llm.generate(prompt)
```

**适用场景**：需要特定格式、需要学习模式

### 1.3.3 Chain of Thought (CoT)

```python
# 引导模型一步步思考
prompt = """
问题：如果有5个苹果，再买3个，现在有多少个？

让我们一步步思考：
1. 原来有5个苹果
2. 又买了3个
3. 5 + 3 = 8
答案是：8个
"""
response = llm.generate(prompt)
```

### 1.3.4 Self-Consistency (CoT-SC)

```python
# 多次采样，取多数答案
answers = []
for _ in range(5):
    ans = llm.generate(prompt_with_cot)
    answers.append(ans)

# 取多数一致的答案
final_answer = most_common(answers)
```

### 1.3.5 Tree of Thoughts (ToT)

```python
# 树状探索多个思考路径
class TreeOfThoughts:
    def __init__(self, llm):
        self.llm = llm
        self.max_depth = 5
        
    def generate_thoughts(self, problem):
        thoughts = [problem]
        
        for depth in range(self.max_depth):
            new_thoughts = []
            for thought in thoughts:
                # 生成多个分支
                branches = self.llm.generate(
                    f"从'{thought}'出发，生成3个不同的思考方向",
                    n=3
                )
                new_thoughts.extend(branches)
            
            # 评估每个分支
            scored = self.evaluate_branches(new_thoughts)
            
            # 选择 Top-K
            thoughts = top_k(scored, k=2)
            
        return self.select_best(thoughts)
```

### 1.3.6 ReAct (Reasoning + Acting)

```python
# 结合推理和行动
def react_agent(question, tools):
    thought = f"思考：{question}"
    history = []
    
    while True:
        # 推理下一步
        action = llm.decide(thought, history, tools)
        
        if action.type == "finish":
            return action.answer
            
        # 执行动作
        result = tools[action.name](action.params)
        
        # 观察结果
        observation = f"执行{action.name}，结果：{result}"
        history.append({"thought": thought, "action": action, "result": result})
        
        thought = f"{thought}。{observation}"
```

---

## 1.4 高级提示词模板

### 1.4.1 代码审查模板

```python
CODE_REVIEW_PROMPT = """
你是一个高级代码审查专家，有15年以上的软件开发经验。

## 上下文 (C)
你正在审查 {language} 代码，这段代码来自 {project_name} 项目。

## 目标 (O)
对以下代码进行全面的技术审查：

```{language}
{code}
```

## 风格 (S)
技术报告风格，包含：
- 发现的问题
- 严重程度 (Critical/High/Medium/Low)
- 修复建议
- 代码评分

## 语气 (T)
专业、建设性、客观

## 受众 (A)
中高级开发工程师

## 响应格式 (R)
请用以下 Markdown 格式输出：

## 审查结果

### 问题清单
| # | 问题 | 严重程度 | 位置 | 建议 |
|---|------|----------|------|------|
| 1 | ... | Critical | ... | ... |

### 总体评分
- 可读性：X/10
- 性能：X/10
- 安全：X/10
- 最佳实践：X/10

### 改进建议
1. ...
2. ...
"""
```

### 1.4.2 数据分析模板

```python
DATA_ANALYSIS_PROMPT = """
你是一个数据科学家，擅长用数据讲故事。

## 上下文
数据集：{dataset_description}
目标变量：{target_variable}
特征：{features}

## 目标
分析数据并回答：{question}

## 风格
数据报告风格，包含：
- 可视化描述
- 统计分析
- 关键发现

## 输出格式
```python
# 请提供可执行的代码
{analysis_code}
```

## 发现模板
### 1. 数据概览
- 总记录数：
- 特征数：
- 缺失值：

### 2. 关键发现
- 发现1：
- 发现2：

### 3. 建议
- 建议1：
"""
```

### 1.4.3 架构设计模板

```python
ARCHITECTURE_PROMPT = """
你是软件架构专家，有20年经验，设计过多个大规模系统。

## 项目背景
项目名称：{project_name}
用户规模：{user_scale}
核心功能：{core_features}
非功能需求：{nfr}

## 目标
设计一个高可用的系统架构

## 风格
架构文档风格，包含：
- 架构图 (ASCII)
- 组件说明
- 技术选型
- 利弊分析

## 输出格式
用 Markdown 输出，包含：

## 架构概览
```
[架构图]
```

## 组件设计
### 1. 前端层
- 技术：
- 职责：

### 2. API Gateway
- 技术：
- 职责：

### 3. 服务层
- 技术：
- 职责：

### 4. 数据层
- 技术：
- 职责：

## 技术选型
| 组件 | 选择 | 理由 |
|------|------|------|
| 计算 | | |
| 存储 | | |
| 缓存 | | |
"""
```

---

## 1.5 实践案例

### 案例 1：简历优化助手

```python
RESUME_OPTIMIZER_PROMPT = """
你是一个资深 HR，拥有 20 年招聘经验，擅长优化简历。

## 候选人背景
- 当前职位：{current_title}
- 工作年限：{years_exp}
- 目标职位：{target_title}
- 行业：{industry}

## 简历内容
{resume_content}

## 目标
优化简历，使其更有竞争力，针对 ATS 系统和 HR 筛选

## 输出格式
请提供：
1. 优化后的简历文本
2. 关键词建议
3. 量化成果建议
"""
```

**使用示例**：

```python
prompt = RESUME_OPTIMIZER_PROMPT.format(
    current_title="初级开发工程师",
    years_exp=3,
    target_title="高级开发工程师",
    industry="互联网",
    resume_content="负责后端开发，使用 Python 和 SQL..."
)
response = llm.generate(prompt)
```

### 案例 2：SQL 生成助手

```python
SQL_GENERATOR_PROMPT = """
你是一个 SQL 专家，擅长写高效的查询。

## 数据库
- 类型：{db_type}
- 版本：{db_version}

## 需求
{user_requested}

## 表结构
{tables}

## 输出
1. SQL 语句
2. 说明
3. 性能提示
"""
```

### 案例 3：API 文档生成

```python
API_DOC_PROMPT = """
你是一个技术文档专家，擅长写清晰的 API 文档。

## 代码
```{language}
{code}
```

## 输出格式
### 接口说明
### 参数
### 返回值
### 示例
### 错误码
"""
```

---

## 1.6 提示词迭代优化

### 优化流程

```
┌─────────────────────────────────────────────────────────────┐
│              提示词优化流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                     │
│  1. 定义目标                                         │
│     ↓                                               │
│  2. 编写初始提示词                                    │
│     ↓                                               │
│  3. 测试输出                                        │
│     ↓                                               │
│  4. 分析问题                                        │
│     ↓                                               │
│  5. 优化提示词 ←──────┐                             │
│     ↓              │     │                            │
│  6. 再次测试        │     │                            │
│     ↓              │     │                            │
│  7. 达到目标？─No──┘                             │
│     ↓                                                  │
│  8. Yes → 完成                                       │
│                                                     │
└─────────────────────────────────────────────────────────────┘
```

### 常见问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 输出太简短 | 没给足上下文 | 增加 Context |
| 格式不对 | 没指定格式 | 明确输出格式 |
| 不够专业 | 没指定角色 | 添加角色描述 |
| 创意不足 | 没引导思考 | 添加 CoT |
| 重复输出 | 没给示例 | Few-shot |

---

## 1.7 参考资源

### GitHub 优秀提示词仓库

| 仓库 | Stars | 主题 |
|------|-------|------|
| [NirDiamant/Prompt_Engineering](https://github.com/NirDiamant/Prompt_Engineering) | - | 提示词技巧 |
| [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) | 10K+ | 提示词指南 |
| [x1xhlol/system-prompts-and-models-of-ai-tools](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools) | 133K | 系统提示词集合 |

### 论文

| 论文 | 主题 |
|------|------|
| Chain of Thought | 思维链推理 |
| Self-Consistency | 自一致性 |
| Tree of Thoughts | 树状思考 |
| Graph of Thoughts | 图状思考 |

### 在线资源

- [Prompt Engineering Guide](https://www.promptingguide.ai)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/preset-chat-constructing-chats-with-tool-use)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

---

*本章节包含实践代码，请参考配套的 `code-examples/` 目录*
*持续更新中...*