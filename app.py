import streamlit as st
from rag_chain import RAGChain
import uuid


st.set_page_config(page_title="Milvus RAG Demo", layout="wide")

# ----------------------
# 初始化：必须全部放最前面，绝对不动
# ----------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "context" not in st.session_state:
    st.session_state["context"] = []
if "rag" not in st.session_state:
    st.session_state["rag"] = RAGChain()
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

st.title("🤖 Milvus RAG Demo")

# ----------------------
# 侧边栏
# ----------------------
with st.sidebar:
    st.subheader("📚 参考资料")
    if st.session_state["context"]:
        for i, ctx in enumerate(st.session_state["context"], 1):
            with st.expander(f"资料 {i}"):
                st.markdown(ctx)
    else:
        st.info("发送问题后显示参考资料")

    if st.button("清空对话"):
        st.session_state["messages"] = []
        st.session_state["context"] = []
        st.rerun()
    st.caption(f"对话轮数: {len(st.session_state['messages'])//2}")

# ----------------------
# 聊天渲染（官方标准结构）
# ----------------------
# 第一步：永远先渲染所有历史
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# 第二步：永远最后放 chat_input
prompt = st.chat_input("请输入问题")
if prompt:
    # 显示用户消息
    st.chat_message(name="user").write(prompt)
    # 保存用户消息到历史
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.spinner("AI思考中..."):
        context = st.session_state["rag"].search(prompt, st.session_state["session_id"])
        ans = st.session_state["rag"].answer(
            context, prompt, st.session_state["session_id"]
        )

        st.session_state["context"] = context

        # write_stream希望接受一个生成器
        res_stream = st.chat_message(name="assistant").write_stream(ans)
        st.session_state["messages"].append(
            {"role": "assistant", "content": res_stream}
        )
        st.rerun()
