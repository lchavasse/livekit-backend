import os
import time
import re
from datetime import datetime, date
from supabase import create_client, Client
import dotenv
import json
from openai import OpenAI
from typing import Optional, Dict, Any, List

from huggingface_hub import InferenceClient

dotenv.load_dotenv()
today = datetime.now().strftime("%Y-%m-%d")
day_of_week = datetime.now().strftime("%A")


url: str = "https://fwcuguulstooyzkkxtvg.supabase.co"
key: str = os.environ.get("SUPABASE_KEY")
hf_token: str = os.environ.get("HUGGINGFACE_API_KEY")
supabase: Client = create_client(url, key)
SUPABASE_USER = os.environ.get("SUPABASE_USER")
SUPABASE_PASSWORD = os.environ.get("SUPABASE_PASSWORD")

response = supabase.auth.sign_in_with_password(
    {"email": SUPABASE_USER, "password": SUPABASE_PASSWORD}
)

# Get the user's UID
user = response.user
if user:
    uid = user.id

client = InferenceClient(api_key=hf_token)

def fetch_user_info(user_id: str = uid) -> Dict[str, Any]:
    """Get user information from the Supabase database."""
    print("###### Fetching user info: ######")
    try:
        response = supabase.table("user_info").select("*").eq("user_id", user_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        return {"error": str(e)}
    
def fetch_user_project(user_id: str = uid, project_ids: List[int] = None) -> Dict[str, Any]:
    """Get user projects from the Supabase database."""
    print("###### Fetching user projects: ######")
    try:
        if project_ids: # get specific projects
            response = supabase.table("user_projects").select("*").eq("user_id", user_id).in_("id", project_ids).execute()
        else: # get all projects
            response = supabase.table("user_projects").select("*").eq("user_id", user_id).execute()
        
        # Convert the response data to a formatted string
        if response.data:
            projects_str = "Here are your projects:\n\n"
            for project in response.data:
                projects_str += f"Project ID: {project['id']}\n"
                projects_str += f"Title: {project['title']}\n"
                projects_str += f"Description: {project['description']}\n"
                if project.get('updates'):
                    projects_str += "Updates:\n"
                    for date, update in project['updates'].items():
                        projects_str += f"- {date}: {update}\n"
                projects_str += "\n"
            return projects_str
        
        return "No projects found."
    except Exception as e:
        return {"error": str(e)}
    
def set_user_info(basic_info: str, recent_projects: List[int] = None, last_conversation: str = None, user_id: str = uid) -> Dict[str, Any]:
    """
    Set all of the basic information for the user.
    - basic_info: string 
    - recent_projects: list[int] - int keys from the user_projects table
    - last_conversation: string
    If you do not know all of the information pass a blank for the missing fields. (ensure the type is correct)
    """
    print("###### Setting user info: ######")
    print(basic_info)
    try:
        # Prepare the data to insert, excluding None values
        data_to_insert = {"user_id": user_id, "basic_info": basic_info}
        if recent_projects is not None:
            print(recent_projects)
            data_to_insert["recent_projects"] = recent_projects
        if last_conversation is not None:
            print(last_conversation)
            data_to_insert["last_conversation"] = last_conversation

        response = supabase.table("user_info").update(data_to_insert).execute()
        return {"success": True, "updated_fields": {"basic_info": basic_info, "recent_projects": recent_projects, "last_conversation": last_conversation}, "data": response.data[0]}
    except Exception as e:
        return {"error": str(e)}
    
def update_user_info(field: str, value: any, user_id: str = uid) -> Dict[str, Any]:
    """
    Update a specific field in the user_info table.
    The available fields are:
    - basic_info: string
    - recent_projects: list[int] - int keys from the user_projects table
    - last_conversation: string
    """
    print("###### Updating user info: ######")
    print(field, value)
    try:
        response = supabase.table("user_info").update({field: value}).eq("user_id", user_id).execute()
        return {"success": True, "updated_fields": {field: value}, "data": response.data[0]}
    except Exception as e:
        return {"error": str(e)}
    
def set_user_project(title: str, description: str, user_id: str = uid) -> Dict[str, Any]:
    """
    Add a new project to the user_projects table.

    ARGS:
    - title: str - the title of the project
    - description: str - the description of the project
    """
    print("###### Setting user project: ######")
    print(title, description)
    try:
        response = supabase.table("user_projects").insert({"title": title, "description": description, "user_id": user_id}).execute()
        return {"success": True, "updated_fields": {"title": title, "description": description}, "data": response.data[0]}
    except Exception as e:
        return {"error": str(e)}

    
def set_project_update(project_id: int, update: str, user_id: str = uid) -> Dict[str, Any]:
    """
    Add an update to a specific project.
    This first fetches the json object from the database, then adds the new update to the object, then upserts the object back into the database.

    ARGS:
    - project_id: int - the id of the project to update
    - update: str - the update to add to the project
    """
    print("###### Setting project update: ######")
    print(project_id, update)
    try:
        # First fetch the project
        response = supabase.table("user_projects").select("updates").eq("id", project_id).eq("user_id", user_id).execute()
        
        # Initialize updates if it doesn't exist
        updates = response.data[0].get("updates", {}) if response.data else {}
        if updates is None:  # If updates is explicitly None in the database
            updates = {}
            
        # Add the new update
        updates[today] = update
        
        # Update the project with the new updates
        response = supabase.table("user_projects").update({"updates": updates}).eq("id", project_id).eq("user_id", user_id).execute()
        return {"success": True, "updated_fields": {"updates": {today: update}}, "data": response.data[0]}
    except Exception as e:
        return {"error": str(e)}
    

def update_user_info(field: str, value: any, user_id: str = uid) -> Dict[str, Any]:
    """
    Update a specific field in the user_info table.
    The available fields are:
    - basic_info: string
    - recent_projects: list[int] - int keys from the user_projects table
    - last_conversation: string
    """
    try:
        response = supabase.table("user_info").update({field: value}).eq("user_id", user_id).execute()
        return {"success": True, "updated_fields": {field: value}, "data": response.data[0]}
    except Exception as e:
        return {"error": str(e)}
    

    

def get_diary_entries(start_date: str, end_date: str = today, user_id: str = uid) -> str:
    """
    Get diary entries from the Supabase database.

    Args:
        start_date: The start date of the diary entries to get. format: YYYY-MM-DD
        end_date: optional, the end date of the diary entries to get. Defaults to today.

    Returns:
        A formatted string containing the diary entries.
    """
    if end_date is None:
        end_date = today
    
    if not start_date:
        return "Error: Start date is required and must be in the format yyyy-mm-dd."
    
    # Validate date format
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    if not re.match(date_pattern, start_date) or not re.match(date_pattern, end_date):
        return "Error: Dates must be in the format yyyy-mm-dd."
    
    print(f"Fetching diary entries from {start_date} to {end_date}")
    # Convert strings to date objects
    try:
        start_date_obj = date.fromisoformat(start_date)
        end_date_obj = date.fromisoformat(end_date)

        response = supabase.table("diary_entries").select("*").eq("user_id", user_id).gte("date", start_date_obj).lte("date", end_date_obj).execute()
        
        if not response.data:
            return f"No diary entries found between {start_date} and {end_date}."
        
        # Format the entries as a string
        entries_str = f"Diary entries from {start_date} to {end_date}:\n\n"
        for entry in response.data:
            print(type(entry))
            try:
                entry = json.loads(entry.get('encrypted_entry', '{}'))
                entries_str += f"Date: {entry.get('date', 'Unknown date')}\n"
                entries_str += f"Entry: {entry.get('content', 'No content')}\n"
                entries_str += f"Rating: {entry.get('rating', 'No rating')}\n"
                entries_str += "\n\n---\n"
            except json.JSONDecodeError:
                continue  # Skip malformed entries
        
        return entries_str
        
    except Exception as e:
        return f"Error: {str(e)}"
    
def conversation_summary(conversation: str) -> Dict[str, Any]:
    """
    Summarise the conversation and update the last_conversation field in the user_info table.
    """
    sys_message = "Take the conversation with the voice assistant and summarise it into a few sentences."
    messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": conversation}]
    response = client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=messages)
    print(response)
    summary = response.choices[0].message.content
    update_user_info(field="last_conversation", value=summary)
    return {"success": True, "updated_fields": {"last_conversation": summary}, "data": summary}

def save_diary_entry(entry: str, rating: int, date: str = today, user_id: str = uid) -> Dict[str, Any]:
    """
    Save a diary entry to the Supabase database.
    """
    print("###### Saving diary entry: ######")
    diary_entry = {
        "date": date,
        "content": entry,
        "rating": rating
    }
    print(date)

    try:
        response = supabase.table("diary_entries").insert({"user_id": user_id, "date": date, "encrypted_entry": diary_entry}).execute()
        return {"success": True, "data": response.data[0]}
    except Exception as e:
        return {"error": str(e)}