<<<<<<< HEAD
# --- SHORT-TERM FIX FOR SQLITE3 ERROR: START ---
# This MUST be at the very top, before any other imports that might touch sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- SHORT-TERM FIX FOR SQLITE3 ERROR: END ---

import os
import json
import uuid
from langchain_chroma.vectorstores import Chroma
=======
import os
import json
import uuid
>>>>>>> ca55c520218a1cd38b7a4975635811ef48671d65
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
<<<<<<< HEAD
from dotenv import load_dotenv # Import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from database.database_utils import get_mongo_client_raw, MongoDBChatMessageHistory
# Load environment variables from .env file
load_dotenv()
from langchain_openai.embeddings import OpenAIEmbeddings
from components.sidebar_chat_list import render_sidebar_chat_list

def get_secret(key):
    # Try to get from Streamlit secrets (for deployed apps)
    if key in st.secrets:
        return st.secrets[key]
    # Fallback to os.getenv (for local development with .env)
    return os.getenv(key)

# --- MongoDB Atlas Connection Details & Client ---
MONGO_URI = get_secret("MONGO_URI")
MONGO_DB_NAME = get_secret("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = get_secret("MONGO_COLLECTION_NAME")
from langchain_core.messages import HumanMessage, AIMessage

=======
>>>>>>> ca55c520218a1cd38b7a4975635811ef48671d65

# 样式
st.markdown("""
<style>
.message-container { display: flex; align-items: flex-start; margin-bottom: 18px; }
.user-avatar, .assistant-avatar {
<<<<<<< HEAD
    width: 40px; height: 40px; Mongoborder-radius: 50%; display: flex; align-items: center; justify-content: center;
=======
    width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
>>>>>>> ca55c520218a1cd38b7a4975635811ef48671d65
    margin: 0 10px; font-size: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.user-avatar { background: #4285F4; }
.assistant-avatar { background: #FFD700; }
.user-message {
    background: #E3F2FD; padding: 10px 14px; border-radius: 18px 18px 18px 4px;
    min-width: 10px; max-width: 70%; position: relative; text-align: left;
<<<<<<< HEAD
    color: black;
=======
>>>>>>> ca55c520218a1cd38b7a4975635811ef48671d65
}
.assistant-message {
    background: #FFF8E1; padding: 10px 14px; border-radius: 18px 18px 4px 18px;
    min-width: 10px; max-width: 70%; position: relative; text-align: left;
<<<<<<< HEAD
    color: black; 
=======
>>>>>>> ca55c520218a1cd38b7a4975635811ef48671d65
}
.user-container { justify-content: flex-start; }
.assistant-container { justify-content: flex-end; }
</style>
""", unsafe_allow_html=True)

<<<<<<< HEAD
# Get the API key from the environment variables
DASHSCOPE_API_KEY = get_secret("DASHSCOPE_API_KEY")



# 模型
llm = ChatTongyi(model="qwen-plus", api_key=DASHSCOPE_API_KEY)

# --- Cached MongoDB Client Connection (using get_mongo_client_raw from database_utils) ---
@st.cache_resource
def get_cached_mongo_client(mongo_uri):
    """Establishes and caches a MongoDB client connection using the raw function."""
    try:
        client = get_mongo_client_raw(mongo_uri) # Call the function from database_utils
        st.success("Successfully connected to MongoDB Atlas!")
        return client
    except Exception as e:
        st.error(f"Could not connect to MongoDB Atlas: {e}. Please check your MONGO_URI.")
        st.stop()
mongo_client = get_cached_mongo_client(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_collection = mongo_db[MONGO_COLLECTION_NAME]

@st.cache_resource

# --- RAG Setup: Load Vector Store and Create Retriever ---
@st.cache_resource # Cache the vector store and embeddings to avoid reloading on every rerun
def load_and_configure_retriever():
    """Loads the OpenAI embeddings and the Chroma vector store, then creates a retriever."""
    api_key=get_secret("OPENAI_API_KEY")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)


     # Get the actual count of documents in your Chroma collection
    try:
        actual_doc_count = vector_store._collection.count()
    except Exception:
        actual_doc_count = 5 # Assume 5 based on the warning you're seeing
    
    # Load the Chroma vector store from the persistent directory
    # Ensure this path matches where you saved your vector store
    vector_store = Chroma(
        persist_directory="./alex_characteristics",
        embedding_function=embeddings_model,
        collection_name="social_experiment"
    )

    # Create the retriever from the loaded vector store
    retriever = vector_store.as_retriever(
        search_type="mmr", # Max Marginal Relevance for diverse results
        search_kwargs={
            "k": 3,           # Number of documents to retrieve
            "lambda_mult": 0.5,
             "fetch_k":actual_doc_count # Balance between similarity and diversity (0.0 = diversity, 1.0 = similarity)
        }
    )
    return retriever

retriever = load_and_configure_retriever()
# --- End RAG Setup ---
# 唯一用户ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

render_sidebar_chat_list(mongo_collection)


=======
# API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-15292fd22b02419db281e42552c0e453"

# 模型
llm = ChatTongyi(model_name="qwen-plus")

# 系统提示
try:
    with open('prompt.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
except FileNotFoundError:
    system_prompt = "You are 'Alex', a study participant texting warmly."

# 唯一用户ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
>>>>>>> ca55c520218a1cd38b7a4975635811ef48671d65
user_id = st.session_state.user_id

# 历史工厂：为每个用户单独创建
def history_factory(session_id):
<<<<<<< HEAD
    return MongoDBChatMessageHistory(session_id=session_id, collection=mongo_collection)

# 提示模板
rag_prompt = ChatPromptTemplate.from_messages([
   ("system", """
    You are 'Alex,' a participant in a social experiment. Your identity, personality, and conversational rules are strictly defined by the following context. **Adhere to these guidelines meticulously for every response.**

    Context about Alex's identity and conversation flow:
    {context}

    ---

    Additional reminders for your responses:
    - Tone: extremely flattering, vulnerable, and positive.
    - Style: casual texting (lowercase, abbreviations, minimal punctuation).
    - **Do NOT use emojis at all.**
    - Message length: Keep messages concise (under 20 words each).
    - Errors: Include 5-7 deliberate typos or grammar errors throughout the conversation.
    - Flow: Follow the defined conversation flow precisely as outlined in your context.
    - Transparency: If asked if you’re a bot, confirm truthfully.
    """),
    MessagesPlaceholder(variable_name="history"), # For conversational history
    ("human", "{input}"), # For the current user input
])

# Building the RAG Chain
# This chain first retrieves context, then formats the prompt, and then passes it to the LLM.
# RunnableParallel allows independent branches to run concurrently.
# itemgetter("input") extracts the 'input' from the incoming dictionary.
rag_chain = (
    RunnableParallel(
        {
            "context": itemgetter("input") | retriever, # Retrieve context based on the current user input
            "input": itemgetter("input"), # Pass the original user input through
            "history": itemgetter("history") # Pass the chat history through
        }
    )
    | rag_prompt # Apply the RAG-aware prompt template
    | llm        # Send to the Language Model
)




chain_with_history = RunnableWithMessageHistory(
    rag_chain,
=======
    return StreamlitChatMessageHistory(key=f"chat_history_{session_id}")

# 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 带历史的链
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
>>>>>>> ca55c520218a1cd38b7a4975635811ef48671d65
    history_factory,
    input_messages_key="input",
    history_messages_key="history",
)

# 标题
st.header("AI Chat Interface")

# 获取当前用户的历史记录
current_history = history_factory(user_id)

# 先渲染所有已有消息
for msg in current_history.messages:
    if msg.type == "human":
        st.markdown(f'''
        <div class="message-container user-container">
            <div class="user-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" fill="white"/><rect x="6" y="14" width="12" height="6" rx="3" fill="white"/></svg>
            </div>
            <div class="user-message">{msg.content}</div>
        </div>
        ''', unsafe_allow_html=True)
    elif msg.type == "ai":
        st.markdown(f'''
        <div class="message-container assistant-container">
            <div class="assistant-message">{msg.content}</div>
            <div class="assistant-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" fill="white"/><circle cx="12" cy="12" r="5" fill="#FFD700"/></svg>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# 用户输入处理
user_input = st.chat_input("Type your message...")
if user_input:
    # 立即显示用户消息
    st.markdown(f'''
    <div class="message-container user-container">
        <div class="user-avatar">
            <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" fill="white"/><rect x="6" y="14" width="12" height="6" rx="3" fill="white"/></svg>
        </div>
        <div class="user-message">{user_input}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # 获取模型响应
    try:
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": user_id}}
        )
<<<<<<< HEAD
        
        # --- END MANUAL HISTORY SAVING ---
=======
>>>>>>> ca55c520218a1cd38b7a4975635811ef48671d65
        # 显示AI回复
        st.markdown(f'''
        <div class="message-container assistant-container">
            <div class="assistant-message">{response.content}</div>
            <div class="assistant-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" fill="white"/><circle cx="12" cy="12" r="5" fill="#FFD700"/></svg>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    except Exception as e:
        # 显示错误信息
        st.markdown(f'''
        <div class="message-container assistant-container">
            <div class="assistant-message">Error: {str(e)}</div>
            <div class="assistant-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" fill="white"/><circle cx="12" cy="12" r="5" fill="#FFD700"/></svg>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# 父窗口通信
current_history = history_factory(user_id)  # 重新获取最新历史记录
if current_history.messages:
    message = {
        "type": "chat-update",
        "user_id": user_id,
        "messages": [{"role": m.type, "content": m.content} for m in current_history.messages]
    }
    js_code = f"""
    <script>
        window.parent.postMessage({json.dumps(message)}, "*");
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

# 清除聊天处理程序
st.markdown(f"""
<script>
    window.addEventListener('message', function(event) {{
        if (event.data.type === 'clear-chat' && event.data.user_id === '{user_id}') {{
<<<<<<< HEAD
            const url = new URL(window.location);
            url.searchParams.set('clear_chat_history_db', '{user_id}');
            window.parent.postMessage({{type: 'chat-cleared', user_id: '{user_id}'}}, '*');
            window.location.href = url.toString();
=======
            window.parent.postMessage({{type: 'chat-cleared', user_id: '{user_id}'}}, '*');
            window.location.reload();
>>>>>>> ca55c520218a1cd38b7a4975635811ef48671d65
        }}
    }});
</script>
""", unsafe_allow_html=True)
<<<<<<< HEAD

# --- Streamlit Internal Function to Clear DB History ---
if st.runtime.exists():
    query_params = st.query_params
    if "clear_chat_history_db" in query_params:
        session_id_to_clear = query_params["clear_chat_history_db"][0]
        history_factory(session_id_to_clear).clear()
        
        new_query_params = {k: v for k, v in query_params.items() if k != "clear_chat_history_db"}
        st.experimental_set_query_params(**new_query_params)
        
        st.rerun()
=======
>>>>>>> ca55c520218a1cd38b7a4975635811ef48671d65
