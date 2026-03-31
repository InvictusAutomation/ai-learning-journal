# 第三部分：AI Agent 开发

> AI Agent 是 AI 时代的核心应用形态 - 让 AI 能够自主完成复杂任务

---

## 3.1 Agent 基础架构

### 什么是 AI Agent？

AI Agent = 大语言模型 (LLM) + 工具 (Tools) + 推理/决策 (Reasoning) + 循环 (Loop)

```
┌─────────────────────────────────────────────────────────────┐
│                  AI Agent 架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │              LLM (大脑)                         │   │
│  │  - 理解意图                                     │   │
│  │  - 生成计划                                     │   │
│  │  - 决策下一步                                   │   │
│  │  - 生成回复                                     │   │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                 │
│  ┌─────────────────────┴───────────────────────────┐   │
│  │              Memory (记忆)                        │   │
│  │  - 短期记忆：当前会话                           │   │
│  │  - 长期记忆：知识库                            │   │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                 │
│  ┌─────────────────────┴───────────────────────────┐   │
│  │              Tools (工具)                      │   │
│  │  - 文件操作    - 网络请求                      │   │
│  │  - 代码执行    - 数据库                        │   │
│  │  - 浏览器    - 消息发送                      │   │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                 │
│  ┌─────────────────────┴───────────────────────────┐   │
│  │              Agent Loop (循环)                   │   │
│  │  - 输入 → 思考 → 行动 → 观察 → 反馈 → ...      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 3.2 Agent 框架对比

### 主流框架

| 框架 | 语言 | 特点 | 适合场景 |
|------|------|------|----------|
| **LangChain** | Python | 全功能、生态丰富 | 生产环境 |
| **LangGraph** | Python | 可视化工作流 | 复杂工作流 |
| **LlamaIndex** | Python | 索��优化 | RAG 应用 |
| **AutoGen** | Python | 多 Agent 协作 | 团队协作 |
| **CrewAI** | Python | 角色扮演 | 多角色场景 |
| **OpenClaw** | TypeScript | 多平台集成 | 个人助手 |

### 框架选择建议

```
┌─────────────────────────────────────────────────────────────┐
│                   框架选择决策树                            │
├─────────────────────────────────────────────────────────────┤
│                                                     │
│  需要什么功能？                                       │
│       │                                              │
│  ┌────┴────┐                                        │
│  ▼         ▼                                          │
│ 个人助手   生产级应用                                  │
│   │         │                                         │
│  OpenClaw  ├─ 简单 RAG → LlamaIndex                   │
│            ├─ 复杂工作流 → LangGraph                  │
│            ├─ 多 Agent → AutoGen/CrewAI               │
│            └─ 企业级 → LangChain                    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 3.3 Agent 实现

### 3.3.1 基础 ReAct Agent

```python
from abc import ABC, abstractmethod
from typing import Any, Callable

class Tool:
    """工具定义"""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def execute(self, **kwargs) -> str:
        return self.func(**kwargs)

class ReActAgent:
    """ReAct 模式的 Agent"""
    
    def __init__(self, llm, tools: list[Tool], max_iterations=10):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.memory = []
    
    def think(self, prompt: str) -> dict:
        """思考下一步行动"""
        
        # 构建系统提示
        system_prompt = f"""你是一个 AI Agent。
        
可用的工具：
{chr(10).join([f"- {t.name}: {t.description}" for t in self.tools.values()])}

你的工作流程：
1. 理解任务
2. 决定使用哪个工具
3. 执行工具
4. 根据结果决定下一步

回复格式（JSON）：
{{
    "thought": "你的思考",
    "action": "工具名或finish",
    "action_input": {{"参数": "值"}},
    "response": "..."
}}
"""
        
        # 添加历史
        history = "\n".join([
            f"Step {i}: {h['thought']} -> {h['action']} = {h['result'][:100]}"
            for i, h in enumerate(self.memory)
        ])
        
        full_prompt = f"{system_prompt}\n\n历史：\n{history}\n\n当前任务：{prompt}"
        
        return self.llm.generate_json(full_prompt)
    
    def run(self, prompt: str) -> str:
        """运行 Agent"""
        
        for i in range(self.max_iterations):
            # 1. 思考
            result = self.think(prompt)
            
            # 2. 记录
            self.memory.append({
                "thought": result.get("thought", ""),
                "action": result.get("action", ""),
                "action_input": result.get("action_input", {}),
                "result": ""
            })
            
            # 3. 判断是否结束
            if result.get("action") == "finish":
                return result.get("response", "")
            
            # 4. 执行工具
            if result.get("action") in self.tools:
                tool = self.tools[result["action"]]
                try:
                    output = tool.execute(**result.get("action_input", {}))
                except Exception as e:
                    output = f"Error: {str(e)}"
                
                self.memory[-1]["result"] = output
                prompt = f"继续执行任务，上一步结果：{output}"
            else:
                return f"未知的工具：{result.get('action')}"
        
        return "达到最大迭代次数"
```

### 3.3.2 Plan-And-Execute Agent

```python
class PlanAndExecuteAgent:
    """计划-执行模式 Agent"""
    
    def __init__(self, llm, executor, tools):
        self.llm = llm
        self.executor = executor
        self.tools = tools
    
    def plan(self, task: str) -> list[dict]:
        """分解任务为步骤"""
        
        prompt = f"""
        把这个任务分解成具体的执行步骤。
        
        任务：{task}
        
        步骤要求：
        1. 每个步骤必须可执行
        2. 步骤之间有依赖关系
        3. 每个步骤有明确的输出
        
        返回 JSON 数组：
        [
            {{"step": 1, "task": "具体任务", "depends": [], "tool": "使用的工具"}},
            {{"step": 2, "task": "具体任务", "depends": [1], "tool": "使用的工具"}}
        ]
        """
        
        return self.llm.generate_json(prompt)
    
    def execute_step(self, step: dict, context: dict) -> Any:
        """执行单个步骤"""
        
        tool_name = step.get("tool")
        
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            return tool.execute(**step.get("params", {}))
        else:
            return self.executor.execute(step["task"])
    
    def run(self, task: str) -> str:
        # 1. 制定计划
        plan = self.plan(task)
        
        context = {}
        results = []
        
        # 2. 执行计划
        for step in plan:
            # 检查依赖
            if step.get("depends"):
                deps_results = [context[d] for d in step["depends"]]
                step["params"]["context"] = deps_results
            
            # 执行
            result = self.execute_step(step, context)
            context[step["step"]] = result
            results.append(result)
        
        # 3. 汇总结果
        return self.summarize(results)
```

### 3.3.3 Multi-Agent 协作系统

```python
class MultiAgentSystem:
    """多 Agent 协作系统"""
    
    def __init__(self, coordinator_llm):
        self.coordinator = coordinator_llm
        self.agents = {}
        self.message_queue = []
    
    def register_agent(self, name: str, agent, role: str, expertise: list[str]):
        """注册 Agent"""
        self.agents[name] = {
            "agent": agent,
            "role": role,
            "expertise": expertise
        }
    
    def route(self, query: str) -> str:
        """路由到合适的 Agent"""
        
        prompt = f"""
        分析这个任务，决定哪个 Agent 处理。
        
        任务：{query}
        
        可用 Agents���
        {chr(10).join([f"- {name}: {info['role']} (专业: {info['expertise']})" 
                      for name, info in self.agents.items()])}
        
        返回 JSON：
        {{"agent": "agent名", "reason": "原因"}}
        """
        
        return self.coordinator.generate_json(prompt)
    
    def run(self, query: str) -> str:
        # 1. 路由
        route_result = self.route(query)
        agent_name = route_result.get("agent")
        
        if agent_name not in self.agents:
            return f"没有合适的 Agent：{route_result.get('reason')}"
        
        # 2. 执行
        agent = self.agents[agent_name]["agent"]
        result = agent.run(query)
        
        return result
```

---

## 3.4 Agent 工作流模式

### 3.4.1 Sequential 模式

```python
class SequentialWorkflow:
    """顺序工作流"""
    
    def __init__(self, steps: list):
        self.steps = steps
    
    def run(self, input_data):
        result = input_data
        for step in self.steps:
            result = step.execute(result)
        return result
```

### 3.4.2 Parallel 模式

```python
import asyncio

class ParallelWorkflow:
    """并行工作流"""
    
    def __init__(self, branches: list):
        self.branches = branches
    
    async def run(self, input_data):
        tasks = [branch.execute(input_data) for branch in self.branches]
        results = await asyncio.gather(*tasks)
        return self.aggregate(results)
```

### 3.4.3 Conditional 模式

```python
class ConditionalWorkflow:
    """条件分支工作流"""
    
    def __init__(self, condition_fn, true_branch, false_branch):
        self.condition_fn = condition_fn
        self.true_branch = true_branch
        self.false_branch = false_branch
    
    def run(self, input_data):
        if self.condition_fn(input_data):
            return self.true_branch.execute(input_data)
        else:
            return self.false_branch.execute(input_data)
```

### 3.4.4 Map-Reduce 模式

```python
class MapReduceWorkflow:
    """Map-Reduce 工作流"""
    
    def __init__(self, map_fn, reduce_fn):
        self.map_fn = map_fn
        self.reduce_fn = reduce_fn
    
    def run(self, items):
        # Map 阶段
        mapped = [self.map_fn(item) for item in items]
        
        # Reduce 阶段
        return self.reduce_fn(mapped)
```

---

## 3.5 Tool 定义最佳实践

### 3.5.1 文件操作工具

```python
class FileTools:
    """文件操作工具集"""
    
    @staticmethod
    @tool(name="file_read", description="读取文件内容")
    def read(path: str) -> str:
        """读取文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    @tool(name="file_write", description="写入文件内容")
    def write(path: str, content: str) -> str:
        """写入文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"已写入 {len(content)} 字符"
    
    @staticmethod
    @tool(name="file_exists", description="检查文件是否存在")
    def exists(path: str) -> bool:
        """检查文件"""
        return os.path.exists(path)
```

### 3.5.2 网络请求工具

```python
class HttpTools:
    """网络请求工具"""
    
    @staticmethod
    @tool(name="http_get", description="发送 GET 请求")
    def get(url: str, params: dict = None) -> dict:
        """GET 请求"""
        response = requests.get(url, params=params)
        return {
            "status": response.status_code,
            "body": response.text[:1000],
            "headers": dict(response.headers)
        }
    
    @staticmethod
    @tool(name="http_post", description="发送 POST 请求")
    def post(url: str, data: dict = None, json: dict = None) -> dict:
        """POST 请求"""
        response = requests.post(url, data=data, json=json)
        return {
            "status": response.status_code,
            "body": response.text[:1000]
        }
```

### 3.5.3 浏览器工具

```python
class BrowserTools:
    """浏览器自动化工具"""
    
    @staticmethod
    @tool(name="browser_navigate", description="导航到 URL")
    def navigate(url: str) -> str:
        """导航"""
        # 使用 playwright 或 selenium
        return f"已导航到 {url}"
    
    @staticmethod
    @tool(name="browser_click", description="点击元素")
    def click(selector: str) -> str:
        """点击"""
        return f"已点击 {selector}"
    
    @staticmethod
    @tool(name="browser_type", description="输入文本")
    def type_(selector: str, text: str) -> str:
        """输入"""
        return f"已在 {selector} 输入 {text}"
    
    @staticmethod
    @tool(name="browser_screenshot", description="截图")
    def screenshot() -> str:
        """截图"""
        return "截图已保存"
```

### 3.5.4 代码执行工具

```python
class CodeTools:
    """代码执行工具"""
    
    @staticmethod
    @tool(name="execute_python", description="执行 Python 代码")
    def execute_python(code: str) -> str:
        """执行 Python"""
        try:
            # 创建执行环境
            local_ns = {}
            exec(code, {}, local_ns)
            return str(local_ns)
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    @tool(name="execute_bash", description="执行 Shell 命令")
    def execute_bash(command: str) -> str:
        """执行 Shell"""
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        return result.stdout or result.stderr
```

---

## 3.6 实践案例

### 案例 1：代码审查 Agent

```python
class CodeReviewAgent:
    """代码审查 Agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [
            Tool("file_read", "读取代码文件", FileTools.read),
            Tool("search_code", "搜索代码", self.search_code)
        ]
        self.agent = ReActAgent(llm, self.tools)
    
    def review(self, code_path: str) -> str:
        prompt = f"""
        审查这个代码文件的质量：
        1. 读取文件
        2. 分析代码质量（可读性、性能、安全）
        3. 给出改进建议
        
        文件路径：{code_path}
        """
        
        return self.agent.run(prompt)
```

### 案例 2：数据处理 Agent

```python
class DataProcessAgent:
    """数据处理 Agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [
            Tool("read_csv", "读取 CSV", self.read_csv),
            Tool("write_csv", "写入 CSV", self.write_csv),
            Tool("execute_python", "处理数据", CodeTools.execute_python)
        ]
        self.agent = ReActAgent(llm, self.tools)
    
    def process(self, input_file: str, output_file: str, instructions: str) -> str:
        prompt = f"""
        处理数据：
        1. 读取 {input_file}
        2. 按照以下要求处理：{instructions}
        3. 保存到 {output_file}
        """
        
        return self.agent.run(prompt)
```

### 案例 3：研究 Agent

```python
class ResearchAgent:
    """研究 Agent - 自动搜索和分析"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [
            Tool("web_search", "搜索网络", web_search),
            Tool("web_fetch", "获取网页", web_fetch),
            Tool("save_note", "保存笔记", self.save_note)
        ]
        self.agent = ReActAgent(llm, self.tools)
    
    def research(self, topic: str) -> str:
        prompt = f"""
        研究主题：{topic}
        
        工作流程：
        1. 搜索相关信息（至少 3 个来源）
        2. 获取关键网页内容
        3. 总结关键发现
        4. 保存研究笔记
        """
        
        return self.agent.run(prompt)
```

### 案例 4：写作 Agent

```python
class WritingAgent:
    """写作 Agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [
            Tool("web_search", "搜索参考资料", web_search),
            Tool("file_read", "读取参考文档", FileTools.read),
            Tool("file_write", "写文档", FileTools.write)
        ]
        self.agent = ReActAgent(llm, self.tools)
    
    def write(self, topic: str, style: str = "technical") -> str:
        prompt = f"""
        撰写关于 {topic} 的文档
        
        风格：{style}
        
        流程：
        1. 搜索相关资料
        2. 整理内容
        3. 生成文档
        """
        
        return self.agent.run(prompt)
```

---

## 3.7 Agent 评估与监控

### 3.7.1 关键指标

```python
class AgentMetrics:
    """Agent 评估指标"""
    
    def __init__(self):
        self.metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "avg_iterations": 0,
            "avg_tokens": 0,
            "tool_usage": {}
        }
    
    def record(self, success: bool, iterations: int, tokens: int, tool_usage: dict):
        """记录运行结果"""
        self.metrics["total_runs"] += 1
        if success:
            self.metrics["successful_runs"] += 1
        else:
            self.metrics["failed_runs"] += 1
        
        # 更新平均
        n = self.metrics["total_runs"]
        self.metrics["avg_iterations"] = (
            self.metrics["avg_iterations"] * (n-1) + iterations
        ) / n
        
        # 工具使用统计
        for tool, count in tool_usage.items():
            self.metrics["tool_usage"][tool] = (
                self.metrics["tool_usage"].get(tool, 0) + count
            )
    
    def get_report(self) -> dict:
        """生成报告"""
        n = self.metrics["total_runs"]
        return {
            "success_rate": self.metrics["successful_runs"] / n,
            "avg_iterations": self.metrics["avg_iterations"],
            "tool_usage": self.metrics["tool_usage"]
        }
```

### 3.7.2 日志记录

```python
import logging
from datetime import datetime

class AgentLogger:
    """Agent 日志"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logger = logging.getLogger("AgentLogger")
        
    def log(self, session_id: str, turn: int, thought: str, 
           action: str, result: str, tokens: int):
        """记录日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "turn": turn,
            "thought": thought,
            "action": action,
            "result": result[:500],
            "tokens": tokens
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
```

---

## 3.8 参考资源

### Agent 框架仓库

| 仓库 | Stars | 主题 |
|------|-------|------|
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 100K+ | Agent 框架 |
| [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) | 101K | LLM 应用 |
| [microsoft/ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) | 55K | Agent 入门 |
| [obra/superpowers](https://github.com/obra/superpowers) | 124K | Agent 工作流 |

### 多 Agent 系统

| 仓库 | Stars | 主题 |
|------|-------|------|
| [AutoGen](https://github.com/microsoft/autogen) | 40K+ | 多 Agent |
| [CrewAI](https://github.com/crewAIInc/crewAI) | 30K+ | 角色 Agent |

---

*本章节包含完整代码实现，请配合代码示例使用*
*持续更新中...*