# mongo_setup.py

import streamlit as st
from database.database_utils import get_mongo_client_raw
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.mongo_client import MongoClient # For explicit type hinting of client

# --- Cached MongoDB Client Connection Setup ---
# This function encapsulates all MongoDB client and collection initialization
@st.cache_resource(show_spinner="Connecting to MongoDB Atlas...") # Show spinner while connecting
def get_mongo_db_connection(mongo_uri: str, db_name: str, collection_name: str) -> tuple[MongoClient, Database, Collection]:
    print("--- DEBUG mongo_setup: Inside get_mongo_db_connection function ---") 
    try:
        client = get_mongo_client_raw(mongo_uri) # Get the raw PyMongo client
        st.success("Successfully connected to MongoDB Atlas!")
        print("--- DEBUG mongo_setup: Successfully connected to MongoDB Atlas! ---")
        
        mongo_db: Database = client[db_name]
        mongo_collection: Collection = mongo_db[collection_name]
        print("--- DEBUG mongo_setup: MongoDB client, db, and collection objects ready. ---")
        
        return client, mongo_db, mongo_collection # Return all three objects
    except Exception as e:
        st.error(f"Could not connect to MongoDB Atlas: {e}. Please check your MONGO_URI.")
        st.stop() # Stop the app if connection fails