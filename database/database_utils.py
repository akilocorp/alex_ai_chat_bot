# database_utils.py

import sys
from pymongo import MongoClient
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pymongo.collection import Collection
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
    def __init__(self, session_id: str, collection: Collection, survey_id:str, agent_id:str, response_id:str):
        self.session_id = session_id
        self.collection = collection
        self.survey_id=survey_id
        self.response_id=response_id
        self.agent_id=agent_id
        logger.info(f"MongoDBChatMessageHistory.__init__: Initializing for session_id: {session_id}")

        # --- MODIFIED: NO UPSERT/INSERTION IN __init__ ---
        # Document creation will now happen in add_message.
        # Here, we only try to load an existing document.
        try:
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
                logger.info(f"__init__: Session '{session_id}' loaded. Messages count: {len(self.conversations.get('messages', []))}")
            else:
                logger.info(f"__init__: Session '{session_id}' not found. Will be created on first message.")
                # self.conversations remains None, which is intended.
        except Exception as e:
            logger.critical(f"__init__: CRITICAL ERROR during MongoDBChatMessageHistory initialization for session '{session_id}': {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize chat history for session {session_id}: {e}") from e

    @property
    @override 
    def messages(self) -> list[BaseMessage]:
        # This property should always reflect the current state in the DB
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

    def add_message(self, message: BaseMessage, response_id, survey_id, agent_id) -> None:
        message_dict = {"type": message.type, "content": message.content}
        logger.info(f"add_message: Attempting to add message for session {self.session_id}: {message_dict['content'][:50]}...")
        
        try:
            # --- MODIFIED: Ensure document exists and is ready for push ---
            # Check if the document exists for this session_id
            existing_doc = self.collection.find_one({"session_id": self.session_id})
            
            if not existing_doc:
                # If document does not exist, create it now (on first message)
                logger.info(f"add_message: Document for session '{self.session_id}' not found. Creating it now.")
                self.collection.insert_one({"session_id": self.session_id, "messages": []})
                existing_doc = self.collection.find_one({"session_id": self.session_id}) # Re-fetch to ensure it's loaded

            # After ensuring document existence, verify 'messages' is an array.
            # This handles cases where 'messages' might be missing or malformed after creation/retrieval.
            if existing_doc and not isinstance(existing_doc.get("messages"), list):
                logger.warning(f"add_message: 'messages' field for session '{self.session_id}' is not an array. Attempting to fix it.")
                self.collection.update_one(
                    {"session_id": self.session_id},
                    {"$set": {"messages": []}}
                )
            
            # Now, push the message (document is guaranteed to exist and 'messages' is an array)
            result = self.collection.update_one(
                {"session_id": self.session_id},
                {"$push": {"messages": message_dict}}
            )
            if result.matched_count > 0:
                logger.info(f"add_message: Message added successfully to {result.matched_count} document for session {self.session_id}.")
            else:
                logger.warning(f"add_message: No document matched for update for session {self.session_id}. This should not happen if previous steps worked.", file=sys.stderr)
        except Exception as e:
            logger.error(f"add_message: ERROR adding message for session {self.session_id}: {e}", exc_info=True)

    def clear(self) -> None:
        logger.info(f"clear: Attempting to clear session: {self.session_id}")
        try:
            result = self.collection.delete_one({"session_id": self.session_id})
            if result.deleted_count > 0:
                logger.info(f"clear: Session {self.session_id} deleted successfully. Deleted count: {result.deleted_count}")
            else:
                logger.warning(f"clear: No document found to delete for session {self.session_id}.")
            self.collection.insert_one({"session_id": self.session_id, "messages": []})
            logger.info(f"clear: Re-created empty document for session: {self.session_id}")
        except Exception as e:
            logger.error(f"clear: ERROR clearing session: {e}", exc_info=True)  