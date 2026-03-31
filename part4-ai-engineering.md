# 第四部分：AI 工程实践

> AI 工程实践是将 AI 技术落地到生产环境的核心能力

---

## 4.1 生产级系统架构

### 4.1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│               AI Platform Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │              API Gateway Layer                     │   │
│  │    - 认证 (Auth)      - 限流 (Rate Limit)       │   │
│  │    - 路由 (Route)     - 日志 (Logging)          │   │
│  │    - 缓存 (Cache)     - 监控 (Monitoring)    │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                 │
│                        ▼                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │            Agent Orchestrator                   │   │
│  │    - 任务分解         - 结果聚合               │   │
│  │    - 调度           - 错误恢复               │   │
│  │    - 重试           - 超时控制               │   │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                 │
│  ┌────────────────────┴────────────────────────┐      │
│  │              Tool Execution Layer            │      │
│  ├─────────────────────────────────────────────┤      │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐      │      │
│  │  │ Code    │ │ File    │ │ HTTP    │      │      │
│  │  │ Executor│ │ Tools   │ │ Client  │      │      │
│  │  └─────────┘ └─────────┘ └─────────┘      │      │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐      │      │
│  │  │Browser  │ │ Vector  │ │ Message │      │      │
│  │  │Driver  │ │ Store   │ │ Sender  │      │      │
│  │  └─────────┘ └─────────┘ └─────────┘      │      │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                 │
│                        ▼                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Data Layer                        │   │
│  │    - Redis (缓存)    - PostgreSQL (数据)     │   │
│  │    - S3 (文件)     - Pinecone (向量)          │   │
│  │    - Neo4j (图)    - Elasticsearch (搜索)      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 4.2 基础设施层实现

### 4.2.1 配置管理

```python
import os
from typing import Optional
from dataclasses import dataclass
import yaml

@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60

@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str
    port: int
    database: str
    user: str
    password: str
    pool_size: int = 10

class Config:
    """统一配置管理"""
    
    def __init__(self, config_file: str = None):
        self.config = {}
        
        if config_file:
            self.load_from_file(config_file)
        else:
            self.load_from_env()
    
    def load_from_env(self):
        """从环境变量加载"""
        self.llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096"))
        )
        
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
    
    def load_from_file(self, path: str):
        """从文件加载"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.llm = LLMConfig(**config.get('llm', {}))
        self.database = DatabaseConfig(**config.get('database', {}))
```

### 4.2.2 连接池管理

```python
import threading
from queue import Queue, Empty
from contextlib import contextmanager

class ConnectionPool:
    """连接池"""
    
    def __init__(self, factory, max_size=10):
        self.factory = factory
        self.max_size = max_size
        self.pool = Queue(max_size)
        self.size = 0
        self.lock = threading.Lock()
    
    @contextmanager
    def get_connection(self):
        """获取连接"""
        conn = None
        
        # 尝试从池中获取
        try:
            conn = self.pool.get_nowait()
        except Empty:
            with self.lock:
                if self.size < self.max_size:
                    conn = self.factory()
                    self.size += 1
                else:
                    # 等待
                    conn = self.pool.get(timeout=30)
        
        try:
            yield conn
        finally:
            if conn:
                self.pool.put(conn)
    
    def close(self):
        """关闭所有连接"""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except Empty:
                break
        self.size = 0
```

---

## 4.3 可观测性

### 4.3.1 指标收集

```python
from prometheus_client import Counter, Histogram, Gauge

# 请求指标
request_count = Counter(
    'ai_requests_total',
    'Total requests',
    ['provider', 'model', 'status']
)

request_duration = Histogram(
    'ai_request_duration_seconds',
    'Request duration',
    ['provider', 'model']
)

# Token 指标
tokens_used = Counter(
    'ai_tokens_used_total',
    'Tokens used',
    ['provider', 'model', 'type']
)

# Agent 指标
active_agents = Gauge(
    'ai_active_agents',
    'Active agents'
)

tool_usage = Counter(
    'ai_tool_usage_total',
    'Tool usage',
    ['tool', 'status']
)

class Metrics:
    """指标收集"""
    
    @staticmethod
    def record_request(provider: str, model: str, 
                     duration: float, status: str):
        request_count.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        request_duration.labels(
            provider=provider,
            model=model
        ).observe(duration)
    
    @staticmethod
    def record_tokens(provider: str, model: str,
                    input_tokens: int, output_tokens: int):
        tokens_used.labels(
            provider=provider,
            model=model,
            type='input'
        ).inc(input_tokens)
        
        tokens_used.labels(
            provider=provider,
            model=model,
            type='output'
        ).inc(output_tokens)
```

### 4.3.2 日志结构化

```python
import json
import logging
from datetime import datetime
from typing import Any

class StructuredLogger:
    """结构化日志"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log(self, level: str, event: str, **kwargs):
        """记录结构化日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            **kwargs
        }
        
        self.logger.info(json.dumps(entry))
    
    def request(self, request_id: str, model: str, 
                duration: float, tokens: int):
        """记录请求"""
        self.log(
            "INFO",
            "ai_request",
            request_id=request_id,
            model=model,
            duration_ms=duration * 1000,
            tokens=tokens
        )
    
    def error(self, request_id: str, error: str, **kwargs):
        """记录错误"""
        self.log(
            "ERROR",
            "ai_error",
            request_id=request_id,
            error=error,
            **kwargs
        )
```

### 4.3.3 分布式追踪

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

class Tracing:
    """分布式追踪"""
    
    @staticmethod
    @tracer.start_as_current_span("ai_request")
    def trace_request(span, model: str, prompt: str):
        span.set_attribute("ai.model", model)
        span.set_attribute("ai.prompt_length", len(prompt))
        
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            span.set_attribute("ai.success", True)
```

---

## 4.4 错误处理与重试

### 4.4.1 错误分类

```python
from enum import Enum

class ErrorType(Enum):
    """错误类型"""
    TRANSIENT = "transient"      # 临时错误，可重试
    RATE_LIMIT = "rate_limit"    # 限流
    AUTH = "auth"                # 认证错误
    VALIDATION = "validation"    # 输入验证
    TIMEOUT = "timeout"          # 超时
    SERVER = "server"           # 服务器错误
    UNKNOWN = "unknown"          # 未知错误

class AIError(Exception):
    """AI 错误"""
    
    def __init__(self, message: str, error_type: ErrorType, 
                 retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.retryable = retryable
```

### 4.4.2 重试策略

```python
import time
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar('T')

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    retryable_errors: tuple = (AIError,)
):
    """指数退避重试"""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_exception = e
                    
                    if not e.retryable or attempt == max_retries:
                        raise
                    
                    # 指数退避
                    delay = base_delay * (exponential_base ** attempt)
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

# 使用示例
class LLMClient:
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def generate(self, prompt: str) -> str:
        try:
            return self.call_api(prompt)
        except RateLimitError as e:
            raise AIError(str(e), ErrorType.RATE_LIMIT, retryable=True)
```

### 4.4.3 熔断器

```python
import time
from threading import Lock

class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, 
                 timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = Lock()
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                else:
                    raise CircuitBreakerOpen()
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        with self.lock:
            self.failure_count = 0
            self.state = "closed"
    
    def on_failure(self):
        with self.lock:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.last_failure_time = time.time()

class CircuitBreakerOpen(Exception):
    pass
```

---

## 4.5 安全

### 4.5.1 输入验证

```python
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class ValidationError:
    """验证错误"""
    field: str
    message: str

class InputValidator:
    """输入验证器"""
    
    MAX_LENGTH = 100000
    MAX_TOKEN_ESTIMATE = 25000
    
    @staticmethod
    def validate_prompt(prompt: str) -> list[ValidationError]:
        """验证提示词"""
        errors = []
        
        if not prompt or not prompt.strip():
            errors.append(ValidationError("prompt", "不能为空"))
        
        if len(prompt) > InputValidator.MAX_LENGTH:
            errors.append(ValidationError(
                "prompt", 
                f"超过最大长度 {InputValidator.MAX_LENGTH}"
            ))
        
        # 检查敏感内容
        sensitive_patterns = [
            (r'system\s*:', "system prompt injection"),
            (r'ignore\s+previous', "prompt injection")
        ]
        
        for pattern, _ in sensitive_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                errors.append(ValidationError(
                    "prompt", 
                    f"检测到潜在注入: {pattern}"
                ))
        
        return errors
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """清理提示词"""
        # 移除可能的注入尝试
        patterns = [
            r'<\|.*?\|>',
            r'\[INST\]',
            r'[/INST]'
        ]
        
        result = prompt
        for pattern in patterns:
            result = re.sub(pattern, '', result)
        
        return result.strip()

### 4.5.2 速率限制

```python
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, requests_per_minute: int = 60,
                 tokens_per_minute: int = 100000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.requests = defaultdict(list)
        self.tokens = defaultdict(list)
    
    def check_request(self, user_id: str) -> bool:
        """检查请求"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # 清理旧记录
        self.requests[user_id] = [
            t for t in self.requests[user_id] 
            if t > minute_ago
        ]
        
        if len(self.requests[user_id]) >= self.requests_per_minute:
            return False
        
        self.requests[user_id].append(now)
        return True
    
    def check_tokens(self, user_id: str, tokens: int) -> bool:
        """检查 token"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        self.tokens[user_id] = [
            t for t in self.tokens[user_id] 
            if t > minute_ago
        ]
        
        total = sum(self.tokens[user_id]) + tokens
        if total > self.tokens_per_minute:
            return False
        
        self.tokens[user_id].append(tokens)
        return True
```

### 4.5.3 审计日志

```python
class AuditLogger:
    """审计日志"""
    
    def __init__(self, storage):
        self.storage = storage
    
    def log_request(self, user_id: str, action: str, 
                  resource: str, result: str):
        """记录请求"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "result": result,
            "ip_address": get_client_ip()
        }
        
        self.storage.save("audit", entry)
    
    def log_data_access(self, user_id: str, data_type: str,
                    record_ids: list):
        """记录数据访问"""
        for record_id in record_ids:
            self.log_request(
                user_id, "read", 
                f"{data_type}:{record_id}", "success"
            )
```

---

## 4.6 部署

### 4.6.1 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 环境变量
ENV PYTHONUNBUFFERED=1

# 启动
CMD ["python", "main.py"]
```

### 4.6.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/ai
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/data

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: ai
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7
    volumes:
      - redisdata:/data

  redisinsight:
    image: redislabs/redisinsight
    ports:
      - "8001:8001"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - ./grafana:/etc/grafana/provisioning/dashboards

volumes:
  pgdata:
  redisdata:
```

### 4.6.3 Kubernetes 部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-platform
  template:
    metadata:
      labels:
        app: ai-platform
    spec:
      containers:
      - name: api
        image: ai-platform:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "2"
            memory: 4Gi
          requests:
            cpu: "500m"
            memory: 1Gi
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-secrets
              key: database-url
---
apiVersion: v1
kind: Service
metadata:
  name: ai-platform
spec:
  selector:
    app: ai-platform
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 4.7 性能优化

### 4.7.1 缓存策略

```python
import hashlib
from functools import wraps

class Cache:
    """简单缓存"""
    
    def __init__(self, redis_client, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[str]:
        return self.redis.get(key)
    
    def set(self, key: str, value: str):
        self.redis.setex(key, self.ttl, value)
    
    def generate_key(self, prompt: str, model: str) -> str:
        """生成缓存键"""
        content = f"{prompt}:{model}"
        return hashlib.md5(content.encode()).hexdigest()

def cached(cache: Cache):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(prompt: str, *args, **kwargs):
            key = cache.generate_key(prompt, kwargs.get('model', 'gpt-4'))
            
            cached_result = cache.get(key)
            if cached_result:
                return cached_result
            
            result = func(prompt, *args, **kwargs)
            cache.set(key, result)
            return result
        
        return wrapper
    return decorator
```

### 4.7.2 批量处理

```python
from typing import List
import asyncio

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, batch_size: int = 10, 
                 max_concurrency: int = 5):
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
    
    async def process_batch(self, items: List[dict], 
                       process_fn) -> List[dict]:
        """批量处理"""
        results = []
        
        # 分批
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i+self.batch_size]
            
            # 并发处理
            tasks = [process_fn(item) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results
```

---

## 4.8 CI/CD

### 4.8.1 GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: pytest --cov
    
    - name: Type check
      run: mypy ai/
    
    - name: Lint
      run: flake8 ai/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Build image
      run: docker build -t ai-platform:${{ github.sha }} .
    
    - name: Deploy to staging
      run: |
        kubectl set image deployment/ai-platform \
        api=ai-platform:${{ github.sha }}
```

---

## 4.9 参考资源

### 监控与可观测性

| 工具 | 用途 | 链接 |
|------|------|------|
| **Prometheus** | 指标 | prometheus.io |
| **Grafana** | 可视化 | grafana.com |
| **Jaeger** | 追踪 | jaegertracing.io |
| **LangSmith** | LLM 监控 | smith.langchain.com |
| **Phoenix** | ML 监控 | phoenix.arize.com |

### 部署

| 工具 | 用途 | 链接 |
|------|------|------|
| **Docker** | 容器 | docker.io |
| **Kubernetes** | 编排 | kubernetes.io |
| **Helm** | K8s 包管理 | helm.sh |

### 安全

| 工具 | 用途 | 链接 |
|------|------|------|
| **OWASP** | 安全标准 | owasp.org |
| **HashiCorp Vault** | 密钥管理 | vaultproject.io |

---

*本章节包含完整的工程实践代码*
*持续更新中...*