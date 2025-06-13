import os
persist_dir = "./alex_characteristics"
if os.path.exists(persist_dir):
    print(f"DEBUG: Found existing directory '{persist_dir}'. Deleting contents for a clean recreation.")
   
else:
    print(f"DEBUG: Directory '{persist_dir}' does not exist. Will create it.")
    os.makedirs(persist_dir, exist_ok=True) # Ensure parent directory exists if it's nested
