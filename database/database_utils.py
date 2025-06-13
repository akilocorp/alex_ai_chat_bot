# database_utils.py

import sys
from pymongo import MongoClient
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pymongo.collection import Collection
from datetime import datetime # Import datetime for timestamps
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("database_utils.py loaded.")

def get_mongo_client_raw(mongo_uri: str):
    logger.info(f"get_mongo_client_raw: Attempting to connect to MongoDB URI: {mongo_uri[:30]}...")
    try:
        client = MongoClient(mongo_uri)
        client.admin.command('ping')
        logger.info("get_mongo_client_raw: MongoDB Ping successful!")
        return client
    except Exception as e:
        logger.error(f"get_mongo_client_raw: ERROR during raw MongoDB connection: {e}", exc_info=True)
        raise ConnectionError(f"Could not connect to MongoDB Atlas: {e}") from e

class MongoDBChatMessageHistory(BaseChatMessageHistory):
    # Pass survey_id, agent_id, response_id as attributes for the class instance
    def __init__(self, session_id: str, collection: Collection, 
                 survey_id: str = "N/A", agent_id: str = "N/A", response_id: str = "N/A"):
        self.session_id = session_id
        self.collection = collection
        self.survey_id = survey_id
        self.agent_id = agent_id
        self.response_id = response_id

        logger.info(f"MongoDBChatMessageHistory.__init__: Initializing for session_id: {session_id}, ResponseID: {response_id}, AgentID: {agent_id}, SurveyID: {survey_id}")

        try:
            # In __init__, we only attempt to *find* the document.
            # Creation (upsert) will happen in add_message on first use.
            self.conversations = self.collection.find_one({"session_id": self.session_id})
            
            if self.conversations:
                # If found, ensure 'messages' field is a list.
                if not isinstance(self.conversations.get("messages"), list):
                    logger.warning(f"__init__: 'messages' field for session '{session_id}' is not an array. Resetting to empty array.")
                    self.collection.update_one(
                        {"session_id": self.session_id},
                        {"$set": {"messages": []}} 
                    )
                    self.conversations = self.collection.find_one({"session_id": self.session_id}) # Re-fetch after fix
                # logger.info(f"__init__: Session '{session_id}' loaded. Messages count: {len(self.conversations.get('messages', []))}")
            else:
                logger.info(f"__init__: Session '{session_id}' not found. Document will be created by add_message on first use.")
                # self.conversations remains None, which is correct here.
        except Exception as e:
            logger.critical(f"__init__: CRITICAL ERROR during MongoDBChatMessageHistory initialization for session '{session_id}': {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize chat history for session {session_id}: {e}") from e

    @property
    @override 
    def messages(self) -> list[BaseMessage]: # type: ignore # 
        doc = self.collection.find_one({"session_id": self.session_id})
        if doc and doc.get("messages") is not None:
            retrieved_messages = []
            for msg_dict in doc["messages"]:
                if msg_dict["type"] == "human":
                    retrieved_messages.append(HumanMessage(content=msg_dict["content"]))
                elif msg_dict["type"] == "ai":
                    retrieved_messages.append(AIMessage(content=msg_dict["content"]))
            return retrieved_messages
        return []

    # MODIFIED: add_message now handles document creation atomically
    
    # FIX START: add_message is now more robust against initial document state
    def add_message(self, message: BaseMessage) -> None:
        message_dict = {"type": message.type, "content": message.content}
        logger.info(f"add_message: Attempting to add message for session {self.session_id}: {message_dict['content'][:50]}...")
        
        try:
            # 1. Define fields to set ONLY IF the document is newly inserted via upsert
            set_on_insert_fields = {
                "messages": [],  # Ensures messages array is created on first insert
                "created_at": datetime.now(), 
                "response_id": self.response_id, 
                "agent_id": self.agent_id,       
                "survey_id": self.survey_id      
            }

            # 2. Find the current document state
            current_doc_state = self.collection.find_one({"session_id": self.session_id})

            if not current_doc_state:
                # Case A: Document does NOT exist (first message of a new chat)
                # Insert it cleanly with all metadata and an empty messages array.
                logger.info(f"add_message: Document for session '{self.session_id}' not found. Inserting new document with initial metadata.")
                initial_document = {
                    "session_id": self.session_id, # Ensure session_id is also set
                    **set_on_insert_fields # Unpack the fields defined for $setOnInsert
                }
                self.collection.insert_one(initial_document)
                logger.info(f"add_message: New document created for session '{self.session_id}'.")
            elif not isinstance(current_doc_state.get("messages"), list):
                # Case B: Document EXISTS, but 'messages' field is missing or not an array.
                # This explicitly fixes a malformed 'messages' field.
                logger.warning(f"add_message: 'messages' field for session '{self.session_id}' is not an array. Fixing it to empty array.")
                self.collection.update_one(
                    {"session_id": self.session_id},
                    {"$set": {"messages": []}}
                )
            
            # 3. Now, the document is guaranteed to exist and its 'messages' field is an array (or will be created correctly).
            # We can safely push the message.
            result = self.collection.update_one(
                {"session_id": self.session_id},
                {"$push": {"messages": message_dict}}
            )

            if result.matched_count > 0:
                logger.info(f"add_message: Message added successfully for session {self.session_id}. Matched: {result.matched_count}.")
            else:
                logger.warning(f"add_message: WARNING: Document for session {self.session_id} not matched for push. This is unexpected after creation/fix!")

        except Exception as e:
            logger.error(f"add_message: CRITICAL ERROR adding message for session {self.session_id}: {e}", exc_info=True)
            # Re-raise the exception to ensure Streamlit logs it properly if it's not caught higher up.
            raise e 
    # FIX END
    def clear(self) -> None:
        logger.info(f"clear: Attempting to clear session: {self.session_id}")
        try:
            result = self.collection.delete_one({"session_id": self.session_id})
            if result.deleted_count > 0:
                logger.info(f"clear: Session {self.session_id} deleted successfully. Deleted count: {result.deleted_count}")
            else:
                logger.warning(f"clear: No document found to delete for session {self.session_id}.")
            # On clear, re-create the document immediately with initial metadata
            # This ensures that if it's cleared and then new messages are added, metadata is still present
            self.collection.insert_one({
                "session_id": self.session_id, 
                "messages": [],
                "created_at": datetime.now(),
                "response_id": self.response_id,
                "agent_id": self.agent_id,
                "survey_id": self.survey_id
            })
            logger.info(f"clear: Re-created empty document for session: {self.session_id}")
        except Exception as e:
            logger.error(f"clear: ERROR clearing session: {e}", exc_info=True)