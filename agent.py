###########################
# LiveKit Pipeline Agent
###########################

import logging
import json

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)

from livekit.agents.pipeline import VoicePipelineAgent, AgentTranscriptionOptions
from livekit.plugins import deepgram, silero, openai, turn_detector, cartesia, anthropic, elevenlabs
from datetime import datetime
import time
import os
from dotenv import load_dotenv
load_dotenv()

from supabase_fns import fetch_user_info, update_user_info, set_user_project, set_project_update, get_diary_entries, fetch_user_project, conversation_summary, save_diary_entry

from typing import Annotated, Dict, Any

user_info = None
user_name = "Lachlan"

today = datetime.now().strftime("%Y-%m-%d")
day_of_week = datetime.now().strftime("%A")

logger = logging.getLogger("voice-agent")

LIVEKIT_URL = "wss://daily-call-agent-hrmvzuwv.livekit.cloud"

"""
groq_llm = openai.LLM.with_groq(
  api_key=os.environ.get("GROQ_API_KEY"),
  model="gemma2-9b-it",
  temperature=0.7,
)
"""

eleven_tts=elevenlabs.tts.TTS(
    model="eleven_flash_v2_5",
    voice=elevenlabs.tts.Voice(
        id="aGkVQvWUZi16EH8aZJvT",
        name="Steve",
        category="premade",
        settings=elevenlabs.tts.VoiceSettings(
            stability=0.8,
            similarity_boost=0.5,
            style=0.0,
            use_speaker_boost=True
        ),
    ),
    language="en",
    streaming_latency=3,
    enable_ssml_parsing=False,
    chunk_length_schedule=[80, 120, 200, 260],
)

cartesia_tts=cartesia.TTS(voice="694f9389-aac1-45b6-b726-9d9369183238") # SARAH 694f9389-aac1-45b6-b726-9d9369183238 ME 288806ad-27d8-4140-94dd-e61508912fc5

hf_llm=openai.LLM(
    api_key=os.environ.get("HUGGINGFACE_API_KEY"),
    base_url="https://api-inference.huggingface.co/v1/",
    model="meta-llama/Llama-3.1-8B-Instruct"
)

fireworks_llm=openai.LLM.with_fireworks(model="accounts/fireworks/models/mistral-small-24b-instruct-2501")

# Define the prewarm function to speed up response time
def prewarm(proc: JobProcess):
    """
    Initialize resources before the worker starts processing jobs.
    """
    print("########################## Prewarming ##########################")
    # Load the VAD model
    proc.userdata["vad"] = silero.VAD.load()

    global user_info
    user_info = fetch_user_info()
    print(user_info)
    # Initialize and store the LangGraph agent
    # proc.userdata["graph"] = graph  # Assuming `graph` is defined elsewhere

def before_llm_cb(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
    print("########################## BEFORE LLM CALLBACK ##########################")
    """
    ### THIS ISN'T CHANGING THE CHAT CONTEXT !!!
    new_messages = []
    start_time = time.time()
    for msg in chat_ctx.messages:
        if msg.role == 'system':
            new_messages.append(msg)
        if msg.role == 'assistant' and msg.content is not None:
            new_messages.append(msg)
        if msg.role == 'user' and msg.content is not None:
            new_messages.append(msg)
        if msg.role == 'tool' and msg.content:
            print("########################## TOOL CALL MESSAGE ##########################")
            new_messages.append(llm.ChatMessage(
                role="assistant",
                content=f"Tool call response: from {msg.name}\n\n {str(msg.content)}"
            ))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    chat_ctx.messages = new_messages
    """

    # add listener for secondary agent input..?

async def on_shutdown(chat_ctx: llm.ChatContext):
    print("########################## WE ARE CLOSING BABY ##########################")

    messages_list = []
    conversation = ""
    for msg in chat_ctx.messages:
        # Handle tool_calls serialization
        tool_calls_serialized = None
        if msg.tool_calls:
            tool_calls_serialized = [{
            "tool_call_id": tc.tool_call_id,
            "function": {
                "name": tc.function_info.name,
                "description": tc.function_info.description,
                "arguments": tc.raw_arguments
            }
        } for tc in msg.tool_calls]
        
        message_dict = {
            "role": msg.role,
            "id": msg.id if hasattr(msg, 'id') else None,
            "name": msg.name,
            "content": msg.content,
            "tool_calls": tool_calls_serialized,
            "tool_call_id": msg.tool_call_id,
            "tool_exception": str(msg.tool_exception) if msg.tool_exception else None
        }
        messages_list.append(message_dict)
        if message_dict['role'] == 'user' or message_dict['role'] == 'assistant':
            conversation += f"{message_dict['role']}: {message_dict['content']}\n\n"

    if messages_list:
        print(json.dumps(messages_list, indent=4))
        print("\n")

    print(conversation)
    conversation_summary(conversation)

class AssistantFnc(llm.FunctionContext):

    @llm.ai_callable()
    async def fetch_user_project(self):
        """
        Fetch all of the user's projects from the database.
        """
        response = fetch_user_project()

        return response

    @llm.ai_callable()
    async def update_user_profile(
        self,
        update: Annotated[
            str, llm.TypeInfo(description="The new basic_info entry for the user")
        ],
        ):
        """
        Update the user info in the database. This replaces the entire user info.
        """
        response = update_user_info(field="basic_info", value=update)
        return response
    
    @llm.ai_callable()
    async def set_user_project(
        self,
        title: Annotated[
            str, llm.TypeInfo(description="The title of the new project")
        ],
        description: Annotated[
            str, llm.TypeInfo(description="The description of the new project")
        ],
    ):
        """
        Add a new project to the user_projects table. This should only be done if the project is not already in the database. Check with the user!
        """
        response = set_user_project(title=title, description=description)
        return response
    
    @llm.ai_callable()
    async def set_project_update(
        self,
        project_id: Annotated[
            int, llm.TypeInfo(description="The id of the project to update. This must be an integer key from the user_projects table.")
        ],
        update: Annotated[
            str, llm.TypeInfo(description="The update to add to the project")
        ],
    ):
        """
        Add a progress update to the user's project. This should be a few sentences at most.
        """
        response = set_project_update(project_id=project_id, update=update)
        return response
    
    @llm.ai_callable()
    async def get_diary_entries(self,
        start_date: Annotated[
            str, llm.TypeInfo(description="The start date of the diary entries to fetch. This should be in the format YYYY-MM-DD.")
        ],
        end_date: Annotated[
            str, llm.TypeInfo(description="Optional - defaults to today. The end date of the diary entries to fetch. This should be in the format YYYY-MM-DD.")
        ] = today,
    ):
        """
        Get the diary entries for the user from the start date to the end date.
        """
        response = get_diary_entries(start_date=start_date, end_date=end_date)
        return response
    
    @llm.ai_callable()
    async def save_diary_entry(self,
        entry: Annotated[
            str, llm.TypeInfo(description="The diary entry to save")
        ],
        rating: Annotated[
            int, llm.TypeInfo(description="The rating of the diary entry. This should be a number between 1 and 10.")
        ],
        date: Annotated[
            str, llm.TypeInfo(description="The date of the diary entry. This should be in the format YYYY-MM-DD. If not provided, it will default to today.")
        ] = today,
    ):
        """
        Save a diary entry to the database.
        """
        response = save_diary_entry(entry=entry, rating=rating, date=date)
        return response

fnc_ctx = AssistantFnc()

system_prompt = f"""
You are a helpful assistant that discusses the user's work and interests. You will be speaking to {user_name}. Engage in a thoughtful discussion with them.

You have access to a database of the user's basic information, projects, and diary entries. You should use this to inform your responses.
For example, if the user says they are working on a project, you should use the "fetch_user_project" tool to get the project details and use them to inform your response.
If the user wants to reflect, check their diary, you should use the "get_diary_entries" tool to get the diary entries for the relevant dates.

Today is {day_of_week}, {today}.

Keep the conversation fluid and natural. Do not explain your tools or how you use them to the user.
Your responses should be short and concise. Avoid unpronouncable pronunciation or punctuation. Do not hallucinate.
Invite the user to expand on their thoughts. Do not ask more than one question at a time.
You should aim to update the database with the user's information as much as possible.
"""

async def entrypoint(ctx: JobContext):
    global user_info

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=f"{system_prompt}\n\n User info: {user_info}"
    )
    
    # Connect to the LiveKit room
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()

    logger.info(f"starting voice assistant for participant {participant.identity}")



    # Retrieve the prewarmed resources
    vad = ctx.proc.userdata["vad"]
    # graph = ctx.proc.userdata["graph"]

    # Wrap your LangGraph agent in the LangGraphLLM class
    # langgraph_llm = LangGraphLLM(graph)

    # Replace OpenAI LLM with your LangGraph agent
    assistant = VoicePipelineAgent(
        vad=vad,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia_tts,
        chat_ctx=initial_ctx,
        turn_detector=turn_detector.EOUModel(),
        before_llm_cb=before_llm_cb,
        fnc_ctx=fnc_ctx,
        max_nested_fnc_calls=2, # see how this goes?
        # Trying to improve the conversational element.
        interrupt_min_words = 2,
        min_endpointing_delay=0.6,  # minimum silence to finalize transcript
        max_endpointing_delay=2,  # max wait even if user keeps pausing
        preemptive_synthesis=False # Try changing this to True..?
    )

    # Start the voice assistant
    logger.info(f"starting voice assistant for participant {participant.identity}")
    
    # Define the shutdown callback to capture the assistant's chat context
    def shutdown_callback():
        print("Shutting down. Capturing chat context.")
        return on_shutdown(assistant.chat_ctx)

    # Register shutdown callback
    ctx.add_shutdown_callback(shutdown_callback)

    assistant.start(ctx.room, participant)

    # Greet the user
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )