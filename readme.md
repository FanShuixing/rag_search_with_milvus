# 🤖 Milvus RAG Demo

一个基于 **Streamlit + LangChain + Milvus + OpenAI** 构建的智能问答系统 Demo，支持多轮对话记忆、Query Rewrite、Rerank，以及参考资料展示。


---

# ✨ 功能特性

## ✅ 基础功能

* Streamlit 聊天界面
* 流式输出（Streaming Response）
* Milvus 向量检索
* OpenAI LLM 回答生成
* 参考资料展示

## ✅ 高级 RAG 能力

* 多轮上下文记忆（Memory）
* Query Rewrite（查询改写）
* Rerank（结果重排序）
* Source Citation（参考资料来源）

---

# 🏗️ 项目架构

```text
用户输入
   ↓
Query Rewrite
   ↓
Milvus Vector Search
   ↓
Rerank
   ↓
LLM Answer
   ↓
Memory 保存对话
```

---

# 📂 项目结构

```text
rag_search_with_milvus/
│
├── app.py                # Streamlit 前端入口
├── rag_chain.py          # RAG 主流程
├── vector_store.py       # Milvus 检索逻辑
├── requirements.txt      # Python 依赖
├── Dockerfile            # Docker 部署配置
└── README.md
```

---

# 🚀 快速开始

## 1. 克隆项目

```bash
git clone https://github.com/your_username/rag_search_with_milvus.git
cd rag_search_with_milvus
```

---

## 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
```

---

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 4. 配置 OpenAI API Key

推荐使用环境变量：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="URL"
```

---

## 5. 启动 Milvus

如果你已经用 Docker 启动 Milvus，可跳过。

参考官方文档：
[https://milvus.io/docs/install_standalone-docker.md](https://milvus.io/docs/install_standalone-docker.md)

---

## 6. 启动 Streamlit 应用

```bash
streamlit run app.py
```

浏览器访问：

```text
http://localhost:8501
```

---

# 🐳 Docker 部署

## 1. 拉取镜像

```bash
docker pull pollypan/my_rag_with_milvus:v1.0
```

---
## 2.创建env文件
```bash
echo "OPENAI_API_KEY=自己的key" > .env
echo "OPENAI_BASE_URL=URL" > .env
```
---
## 3. 运行容器

```bash
docker compose up -d
```

---

## 4. 访问应用

```text
http://localhost:8501
```

---

# ⚠️ 部署注意事项

如果 Milvus 也是 Docker 启动：

* 不要直接使用 `localhost`
* Docker 容器中建议连接：

```python
host="host.docker.internal"
```

或使用 docker-compose 服务名。

---

# 🧠 技术栈

* Streamlit
* LangChain
* OpenAI API
* Milvus
* Sentence Transformers
* PyTorch
* NumPy
* Docker

---

# 🔧 已实现优化能力说明

## Query Rewrite

根据历史对话，将用户问题改写成更适合检索的完整问题。

例如：

```text
原问题：它的优点是什么？
改写后：Transformer 的优点是什么？
```

---

## Rerank

对检索结果重新排序，把最相关内容优先提供给 LLM。

---

## Memory

支持多轮连续对话，保留上下文。

---

# 📌 后续可扩展方向

* Hybrid Search
* Agent Tool Calling
* 多用户 Session 管理
* 对话总结 Memory
* Web Search 集成
* 文档上传解析

---

# 🙌 致谢

感谢以下开源项目：

* LangChain
* Streamlit
* Milvus
* OpenAI
* Sentence Transformers

