import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- SHORT-TERM FIX FOR SQLITE3 ERROR: START ---
# This MUST be at the very top, before any other imports that might touch sqlite3

from pymongo.database import Database  # NEW IMPORT for type hinting Database
from pymongo.collection import Collection # Ensure this is imported for Collection type hinting (might already be there)
from langchain_core.utils.utils import convert_to_secret_str
import os
import json
import uuid
from langchain_chroma.vectorstores import Chroma
from rag.retriever import get_retriever
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from dotenv import load_dotenv # Import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from database.database_utils import get_mongo_client_raw, MongoDBChatMessageHistory
from database.mongo_setup import get_mongo_db_connection
# Load environment variables from .env file
from langchain_openai.embeddings import OpenAIEmbeddings
folder_path = "./alex_characteristics"
if os.path.exists(folder_path):
    if os.path.isdir(folder_path):
        # List all files and directories inside it
        contents = os.listdir(folder_path)
        if contents:
            for item in contents:
                logger.info(item)
        else:
            logger.info(f"The folder '{folder_path}' is empty.")
    else:
        logger.info(f"Error: '{folder_path}' exists but is not a directory.")
else:
    logger.info(f"Error: The folder '{folder_path}' does not exist.")

print("--- End of contents ---")
load_dotenv()
# from components.sidebar_chat_list import render_sidebar_chat_list
def get_query_param_value(param_key: str, default_val: str = "N/A") -> str:
    """Safely retrieves a query parameter value, handling None or empty list cases."""
    param_list = st.query_params.get(param_key)
    if param_list is None: # Parameter not found in the URL at all
        return default_val
    if not param_list: # Parameter found, but its list of values is empty (e.g., ?param=)
        return default_val
    return param_list[0] # Return the first value from the list


def get_secret(key):
    # Try to get from Streamlit secrets (for deployed apps)
    if key in st.secrets:
        return st.secrets[key]
    # Fallback to os.getenv (for local development with .env)
    return os.getenv(key)

# --- MongoDB Atlas Connection Details & Client ---
DASHSCOPE_API_KEY = get_secret("DASHSCOPE_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
MONGO_URI_VAL = get_secret("MONGO_URI") # Use a temporary variable for clarity during check
MONGO_DB_NAME_VAL = get_secret("MONGO_DB_NAME") # Use a temporary variable for clarity during check
MONGO_COLLECTION_NAME_VAL = get_secret("MONGO_COLLECTION_NAME") # Use a temporary variable for clarity during check

print(f"--- DEBUG MAIN: API Keys loaded: DASHSCOPE_API_KEY starts with {DASHSCOPE_API_KEY[:5] if DASHSCOPE_API_KEY else 'N/A'}, OPENAI_API_KEY starts with {OPENAI_API_KEY[:5] if OPENAI_API_KEY else 'N/A'} ---")


# --- IMPORTANT FIX: Add explicit checks for all required secret values ---
# This ensures that if a secret is missing (returns None), the app stops cleanly.
if not isinstance(DASHSCOPE_API_KEY, str) or not DASHSCOPE_API_KEY:
    st.error("DASHSCOPE_API_KEY not found or invalid. Please set it in your .env file locally or in Streamlit Cloud secrets.")
    st.stop()
if not isinstance(OPENAI_API_KEY, str) or not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found or invalid. This is needed for embeddings. Please set it.")
    st.stop()
if not isinstance(MONGO_URI_VAL, str) or not MONGO_URI_VAL:
    st.error("MONGO_URI not found or invalid. Please set it in your .env file locally or in Streamlit Cloud secrets.")
    st.stop()
if not isinstance(MONGO_DB_NAME_VAL, str) or not MONGO_DB_NAME_VAL:
    st.error("MONGO_DB_NAME not found or invalid. Please set it in your .env file locally or in Streamlit Cloud secrets.")
    st.stop()
if not isinstance(MONGO_COLLECTION_NAME_VAL, str) or not MONGO_COLLECTION_NAME_VAL:
    st.error("MONGO_COLLECTION_NAME not found or invalid. Please set it in your .env file locally or in Streamlit Cloud secrets.")
    st.stop()


from langchain_core.messages import HumanMessage, AIMessage

#Change Api Keys to Secret Str
Dashscope_api=convert_to_secret_str(DASHSCOPE_API_KEY)

# 样式
st.markdown("""
<style>
.message-container { display: flex; align-items: flex-start; margin-bottom: 18px; }
.user-avatar, .assistant-avatar {
    width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    margin: 0 10px; font-size: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.user-avatar { background: #4285F4; }
.assistant-avatar { background: #FFD700; }
.user-message {
    background: #E3F2FD; padding: 10px 14px; border-radius: 18px 18px 18px 4px;
    min-width: 10px; max-width: 70%; position: relative; text-align: left;
    color: black;
}
.assistant-message {
    background: #FFF8E1; padding: 10px 14px; border-radius: 18px 18px 4px 18px;
    min-width: 10px; max-width: 70%; position: relative; text-align: left;
    color: black; 
}
.user-container { justify-content: flex-start; }
.assistant-container { justify-content: flex-end; }
</style>
""", unsafe_allow_html=True)





# 模型
llm = ChatTongyi(model="qwen-plus", api_key=Dashscope_api)

mongo_client, mongo_db, mongo_collection=get_mongo_db_connection(mongo_uri=MONGO_URI_VAL, db_name=MONGO_DB_NAME_VAL, collection_name=MONGO_COLLECTION_NAME_VAL)

#Get the parameters from the link
response_id=get_query_param_value("responseId")
agent_id=get_query_param_value("agentId")
survey_id=get_query_param_value("surveyId")


open_ai_key=convert_to_secret_str(OPENAI_API_KEY)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=open_ai_key)
try:
        vector_store = Chroma(
            persist_directory='./alex_characteristics',
            embedding_function=embeddings_model,
            collection_name='Social_experiment'
        )
        actual_doc_count = vector_store._collection.count()
        print(f"--- DEBUG retriever_setup: Vector store loaded. Actual doc count: {actual_doc_count} ---") 
except Exception as e:
        st.error(f"Could not load Chroma vector store. Ensure ./alex_characteristics directory exists and is accessible. Error: {e}")
        st.stop()
        
RETRIEVAL_K = min(3, actual_doc_count) # Set k to 3, but not more than available
print(f"--- DEBUG retriever_setup: Retrieval K set to {RETRIEVAL_K} ---") 

retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": RETRIEVAL_K,           
            "lambda_mult": 0.5,         
            "fetch_k": actual_doc_count # Fetch up to actual count to avoid warnings
        }
    )

# 唯一用户ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())




user_id = st.session_state.user_id

# 历史工厂：为每个用户单独创建
def history_factory(session_id):
    return MongoDBChatMessageHistory(
        session_id=session_id, 
        collection=mongo_collection,
        response_id=response_id,
        agent_id=agent_id,
        survey_id=survey_id
           )

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
    history_factory,
    input_messages_key="input",
    history_messages_key="history",
)

# 标题
st.header("Alex")

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
            const url = new URL(window.location);
            url.searchParams.set('clear_chat_history_db', '{user_id}');
            window.parent.postMessage({{type: 'chat-cleared', user_id: '{user_id}'}}, '*');
            window.location.href = url.toString();
        }}
    }});
</script>
""", unsafe_allow_html=True)

# --- Streamlit Internal Function to Clear DB History ---
if st.runtime.exists():
    query_params = st.query_params
    if "clear_chat_history_db" in query_params:
        session_id_to_clear = query_params["clear_chat_history_db"][0]
        history_factory(session_id_to_clear).clear()
        
        new_query_params = {k: v for k, v in query_params.items() if k != "clear_chat_history_db"}
        st.experimental_set_query_params(**new_query_params)
        
        st.rerun()

