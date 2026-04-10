from vector_store import VectorStore
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from sentence_transformers import CrossEncoder
from langchain_core.prompts import MessagesPlaceholder

"""
RAG主流程:
记忆功能✅


已实现✅
- Streamlit 聊天 UI
- 流式输出
- Milvus 检索
- LangChain Memory
- 多轮对话
- 基础 RAG
- Query Rewrite：把用户当前不完整的问题【它怎么存储数据】，结合聊天历史改写成【milvus如何存储数据】，然后再送给milvus做检索
- Rerank
"""


class RAGChain:
    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.model = ChatOpenAI(name="gpt-3.5-turbo")  # type: ignore
        self.chat_history_store = {}
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个专业问答小助手，请根据参考资料回答问题"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "参考资料:{context}\n用户问题:{query}"),
            ]
        )
        # Memory
        self.chain = self.prompt_template | self.model
        self.conversation_chain = RunnableWithMessageHistory(
            self.chain,
            get_session_history=self._get_history,
            input_messages_key="query",
            history_messages_key="chat_history",
        )
        # 增加rerank排序
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rewrite_query(self, query: str, session_id: str) -> str:
        """
        根据聊天历史，把当前问题改写成完整问题
        """
        history = self._get_history(session_id)
        history_text = ""
        for msg in history:
            if isinstance(msg, tuple):
                role, content = msg
                role = "用户" if role == "human" else "助手"
            else:

                role = "用户" if msg.type == "human" else "助手"
                content = msg.content
            history_text += f"{role}:{content}\n"

        rewrite_prompt = f"""
        请根据历史对话，将用户当前问题改写成完整、明确、适合知识库检索的问题。

        【历史对话】
        {history_text}

        【当前问题】
        {query}

        【改写要求】
        1. 保持原意
        2. 补全代词指代（它、这个、上述内容等）
        3. 输出最终改写结果
        4. 不要解释

        改写后的问题：
        """

        resp = self.model.invoke(rewrite_prompt)
        new_query = resp.content
        print("原问题:", query)
        print("改写后:", new_query)
        return new_query

    def search(self, query, session_id):
        """
        检索相关文档
        """
        # Query rewirte
        new_query = self.rewrite_query(query, session_id)
        # milvus召回
        context_list = self.vector_store.search(new_query)
        print("-" * 20, "rerank前：\n")
        for i, doc in enumerate(context_list, 1):
            print(f"{i}. {doc[:100]}")
        # Rerank
        rerank_context_lsit = self.rerank(query, context_list)
        print("-" * 20, "rerank后\n")
        for i, doc in enumerate(rerank_context_lsit, 1):
            print(f"{i}. {doc[:100]}")
        return rerank_context_lsit

    def _get_history(self, session_id):
        if session_id not in self.chat_history_store:
            self.chat_history_store[session_id] = InMemoryChatMessageHistory()
        return self.chat_history_store[session_id]

    def answer(self, context_list: list[str], query: str, session_id: str):
        # 处理context
        context = ""
        for each in context_list:
            context += "参考资料" + each + "\n"
        print("------------------------context:\n", context)

        for chunk in self.conversation_chain.stream(
            {"context": context, "query": query},
            config={"configurable": {"session_id": session_id}},
        ):
            if chunk.content:
                yield chunk.content

    def rerank(self, query, context_list, top_n=3):
        """
        对检索结果重新排序
        """

        pairs = [[query, doc] for doc in context_list]
        scores = self.reranker.predict(pairs)

        ranked_docs = sorted(
            zip(context_list, scores), key=lambda x: x[1], reverse=True
        )

        reranked_context = [doc for doc, _ in ranked_docs[:top_n]]
        return reranked_context


if __name__ == "__main__":
    rag = RAGChain()
    print("AI回复:\n")
