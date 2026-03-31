# 第二部分：RAG 高级技术

> RAG (Retrieval-Augmented Generation) 是 AI 应用的核心技术 - 让 AI 拥有最新、最准的知识

---

## 2.1 RAG 基础架构

### 经典 RAG 流程

```
┌─────────────────────────────────────────────────────────────┐
│                  经典 RAG 流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ 文档集合  │───▶│  分块   │───▶│ 向量化  │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│                                          │             │
│                                          ▼             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ 用户查询 │───▶│ 向量化  │───▶│ 向量检索 │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│                                          │             │
│                                          ▼             │
│                                   ┌──────────┐       │
│                                   │  上下文  │       │
│                                   │  组装    │       │
│                                   └──────────┘       │
│                                          │             │
│                                          ▼             │
│                                   ┌──────────┐       │
│                                   │  LLM    │       │
│                                   │  生成   │       │
│                                   └──────────┘       │
│                                          │             │
│                                          ▼             │
│                                   ┌──────────┐       │
│                                   │  最终   │       │
│                                   │  答案   │       │
│                                   └──────────┘       │
└─────────────────────────────────────────────────────┘
```

---

## 2.2 基础 RAG 实现

### 2.2.1 简单 RAG

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SimpleRAG:
    """最简单的 RAG 实现"""
    
    def __init__(self, documents, embed_model="text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=embed_model)
        
        # 1. 文档分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # 2. 向量存储
        self.vectorstore = FAISS.from_documents(
            splits,
            self.embeddings
        )
        
        # 3. 检索器
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
    def query(self, question):
        # 1. 检索相关文档
        docs = self.retriever.invoke(question)
        
        # 2. 组装上下文
        context = "\n\n".join([d.page_content for d in docs])
        
        # 3. 生成答案
        prompt = f"""基于以下参考文档回答问题。

参考文档：
{context}

问题：{question}

请给出准确、详细的回答。"""
        
        return llm.generate(prompt), docs
```

**使用示例**：

```python
# 加载文档
loader = TextLoader("data/article.txt")
documents = loader.load()

# 初始化 RAG
rag = SimpleRAG(documents)

# 查询
answer, docs = rag.query("什么是机器学习？")
```

### 2.2.2 带来源标注的 RAG

```python
class SourceAttributedRAG(SimpleRAG):
    """带来源标注的 RAG"""
    
    def __init__(self, documents, embed_model="text-embedding-3-small"):
        super().__init__(documents, embed_model)
        
    def query(self, question):
        docs = self.retriever.invoke(question)
        
        # 构建带来源的上下文
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"[{i+1}] 来源: {source}\n内容: {doc.page_content}"
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""基于以下参考文档回答问题。

参考文档：
{context}

问题：{question}

要求：
1. 用中文回答
2. 在回答中标注来源编号
"""
        
        return llm.generate(prompt), docs
```

---

## 2.3 高级 RAG 技术

### 2.3.1 HyDE (Hypothetical Document Embedding)

```python
class HyDERAG:
    """
    HyDE - 假设文档嵌入
    核心思想：让 LLM 生成一个"假设答案"，用这个假设答案去检索
    """
    
    def __init__(self, documents, embeddings):
        # 基础索引
        self.base_vectorstore = FAISS.from_documents(
            documents,
            embeddings
        )
        self.retriever = self.base_vectorstore.as_retriever()
        
    def generate_hypothetical_doc(self, query, llm):
        """生成假设文档"""
        prompt = f"""
        生成一个理想回答以下问题的文档。
        直接给出答案，不需要解释。
        
        问题：{query}
        """
        return llm.generate(prompt)
    
    def retrieve(self, query, llm):
        # 1. 生成假设文档
        hypo_doc = self.generate_hypothetical_doc(query, llm)
        
        # 2. 用假设文档检索
        results = self.retriever.invoke(hypo_doc)
        
        return results
```

### 2.3.2 Reranking (两阶段检索)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class RerankingRAG:
    """两阶段检索：粗排 + 精排"""
    
    def __init__(self, documents, embeddings):
        # 第一阶段：向量检索
        self.raw_vectorstore = FAISS.from_documents(
            documents,
            embeddings
        )
        self.raw_retriever = self.raw_vectorstore.as_retriever(k=20)
        
        # 第二阶段：Cross-Encoder 重排
        self.reranker = CrossEncoderReranker(
            HuggingFaceCrossEncoder(model="BAAI/bge-reranker-base"),
            top_k=5
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.raw_retriever
        )
        
    def query(self, query):
        # 两阶段检索
        docs = self.compression_retriever.invoke(query)
        return docs
```

### 2.3.3 Contextual Chunk Headers

```python
class ContextualChunkRAG:
    """为每个块添加上下文标题"""
    
    def __init__(self, documents, embeddings, llm):
        self.llm = llm
        
        # 1. 为每个块生成上下文
        enriched_docs = []
        for doc in documents:
            # 生成块标题
            header = self.llm.generate(f"""
                为以下内容生成一个简短描述性标题（5-10字）：
                
                {doc.page_content[:200]}
            """)
            
            # 添加上下文
            enriched_content = f"""【主题：{header}】\n{doc.page_content}"""
            enriched_docs.append(
                Document(page_content=enriched_content, metadata=doc.metadata)
            )
        
        # 2. 向量化
        self.vectorstore = FAISS.from_documents(enriched_docs, embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
    def query(self, query):
        return self.retrier.invoke(query)
```

### 2.3.4 Adaptive Retrieval

```python
class AdaptiveRAG:
    """自适应的检索策略"""
    
    def __init__(self, documents, embeddings):
        # 多种检索器
        self.similarity_retriever = FAISS(...).as_retriever(search_type="similarity")
        self.mmr_retriever = FAISS(...).as_retriever(search_type="mmr")
        self.similarity_threshold = FAISS(...).as_retriever(search_type="similarity_threshold")
        
    def determine_strategy(self, query, llm):
        """根据查询类型决定检索策略"""
        
        # 分析查询
        analysis = llm.analyze(f"""
            分析这个查询的特点：
            {query}
            
            返回 JSON：
            {{"type": "factual|complex|exploratory", "entities": number}}
        """)
        
        if analysis["type"] == "factual":
            return self.similarity_retriever, {"k": 3}
        elif analysis["type"] == "complex":
            return self.similarity_retriever, {"k": 5}
        else:
            return self.mmr_retriever, {"k": 10, "fetch_k": 20}
    
    def query(self, query):
        strategy, params = self.determine_strategy(query)
        return strategy.invoke(query, **params)
```

---

## 2.4 Graph RAG (知识图谱增强)

### 2.4.1 基础 Graph RAG

```python
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from kgcve import KeyphraseVectorExtraction

class GraphRAG:
    """知识图谱增强的 RAG"""
    
    def __init__(self, documents, neo4j_uri, neo4j_user, neo4j_password):
        # 1. 初始化知识图谱
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )
        
        # 2. 提取知识图谱
        self.extract_graph(documents)
        
        # 3. 创建向量索引
        self.vectorstore = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            graph=self.graph,
            node_label="Document",
            text_property="content",
            embedding_property="embedding"
        )
        
    def extract_graph(self, documents):
        """从文档提取知识图谱"""
        for doc in documents:
            # 提取实体
            entities = self.extract_entities(doc.page_content)
            
            # 提取关系
            relations = self.extract_relations(doc.page_content)
            
            # 存入图谱
            for entity in entities:
                self.graph.query(f"""
                    MERGE (e:Entity {{name: $name, type: $type}})
                """, {"name": entity["name"], "type": entity["type"]})
            
            for relation in relations:
                self.graph.query(f"""
                    MATCH (a:Entity {name: $source})
                    MATCH (b:Entity {name: $target})
                    MERGE (a)-[r:RELATES {{type: $type}}]->(b)
                """, relation)
    
    def query(self, query):
        # 1. 向量检索
        vector_results = self.vectorstore.similarity_search(query, k=5)
        
        # 2. 图谱检索
        graph_results = self.graph.query(f"""
            MATCH (e:Entity)<-[:RELATED]-(d:Document)
            WHERE e.name CONTAINS $query
            RETURN d
        """, {"query": query})
        
        # 3. 融合结果
        combined_results = self.fuse_results(vector_results, graph_results)
        
        return self.generate(combined_results, query)
```

---

## 2.5 Agentic RAG

### 2.5.1 自主 RAG Agent

```python
class AgenticRAG:
    """Agent 驱动的 RAG - 可自我反思和迭代"""
    
    def __init__(self, documents, llm, embeddings):
        self.llm = llm
        self.documents = documents
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
    def evaluate_relevance(self, question, docs):
        """评估文档相关性"""
        prompt = f"""
        评估这些文档是否能回答问题。
        
        问题：{question}
        
        文档：
        {chr(10).join([f"[{i}] {d.page_content[:200]}" for i, d in enumerate(docs)])}
        
        返回 JSON：
        {{"relevant": true/false, "score": 0-10, "reason": "..."}}
        """
        return self.llm.generate_json(prompt)
    
    def should_retrieve_more(self, question):
        """判断是否需要继续检索"""
        prompt = f"""
        判断这个问题是否需要外部知识。
        
        问题：{question}
        
        如果需要最新/特定知识，返回 true；否则返回 false。
        """
        return self.llm.generate_bool(prompt)
    
    def run(self, question, max_iterations=3):
        history = []
        
        for i in range(max_iterations):
            # 1. 检索
            docs = self.retriever.invoke(question)
            
            # 2. 评估相关性
            eval = self.evaluate_relevance(question, docs)
            
            if eval["relevant"] or i == max_iterations - 1:
                # 生成答案
                answer = self.generate_answer(question, docs)
                return answer, docs
            
            # 3. 需要更多检索
            history.append(docs)
            # 优化查询
            question = self.optimize_query(question, history)
```

---

## 2.6 评估指标

### RAG 评估框架

```python
class RAGEvaluator:
    """RAG 评估器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate_faithfulness(self, question, answer, retrieved_docs):
        """评估答案是否来自检索内容"""
        prompt = f"""
        评估答案是否来自提供的文档。
        
        问题：{question}
        答案：{answer}
        
        文档：
        {chr(10).join([d.page_content for d in retrieved_docs])}
        
        评分：1-5
        理由：
        """
        return self.llm.generate(prompt)
    
    def evaluate_answer_relevance(self, question, answer):
        """评估答案与问题的相关度"""
        prompt = f"""
        评估答案与问题的相关程度。
        
        问题：{question}
        答案：{answer}
        
        评分：1-5
        """
        return self.llm.generate(prompt)
    
    def evaluate(self, question, answer, retrieved_docs):
        """综合评估"""
        return {
            "faithfulness": self.evaluate_faithfulness(question, answer, retrieved_docs),
            "answer_relevance": self.evaluate_answer_relevance(question, answer),
            "context_precision": len(retrieved_docs) / 10,
        }
```

---

## 2.7 实践案例

### 案例 1：企业文档问答系统

```python
class EnterpriseDocQA:
    """企业级文档问答系统"""
    
    def __init__(self, doc_paths):
        # 加载文档
        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".md": TextLoader
        }
        
        all_docs = []
        for path in doc_paths:
            ext = os.path.splitext(path)[1]
            loader = loaders.get(ext, TextLoader)(path)
            all_docs.extend(loader.load())
        
        # 初始化 RAG
        self.rag = SimpleRAG(all_docs)
    
    def query(self, question):
        return self.rag.query(question)
```

### 案例 2：多源聚合问答

```python
class MultiSourceRAG:
    """多数据源聚合问答"""
    
    def __init__(self):
        self.rags = {
            "internal": InternalDocRAG(),
            "web": WebRAG(),
            "database": DatabaseRAG()
        }
    
    def query(self, question):
        # 并行检索
        results = parallel(
            {source: rag.query(question) for source, rag in self.rags.items()}
        )
        
        # 融合
        fused = self.fuse(results)
        
        return self.generate(fused, question)
```

### 案例 3：客服机器人

```python
class CustomerServiceBot:
    """客服 RAG 机器人"""
    
    def __init__(self, faq_docs, product_docs):
        self.faq_rag = RerankingRAG(faq_docs)
        self.product_rag = SimpleRAG(product_docs)
        self.product_info = ProductDatabase()
    
    def query(self, user_question):
        # 1. 判断类型
        q_type = self.classify(user_question)
        
        if q_type == "faq":
            return self.faq_rag.query(user_question)
        elif q_type == "product":
            # 产品查询
            return self.product_rag.query(user_question)
        else:
            # 转人工
            return "这个问题我需要转接人工客服"
```

---

## 2.8 参考资源

### RAG 技术仓库

| 仓库 | Stars | 主题 |
|------|-------|------|
| [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | 26K | 34种RAG技术 |
| [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) | 101K | RAG应用集合 |
| [upstash/context7](https://github.com/upstash/context7) | 51K | 代码文档RAG |

### 向量数据库

| 数据库 | 特点 | 链接 |
|--------|------|------|
| Pinecone | 全托管 | pinecone.io |
| Milvus | 开源 | milvus.io |
| Qdrant | 轻量 | qdrant.tech |
| Weaviate | 语义 | weaviate.io |

### 框架文档

- [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/)
- [LlamaIndex](https://docs.llamaindex.ai/)

---

*本章节包含完整代码实现，请配合代码示例使用*
*持续更新中...*