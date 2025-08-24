from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import logging
import time
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
load_dotenv()
# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Isooko API", version="1.0.0")


ASSISTANT_API_KEY = os.getenv("Apikey")
ASSISTANT_ID = os.getenv("assistantId")


client = openai.OpenAI(api_key=ASSISTANT_API_KEY)


class MessageRequest(BaseModel):
    message: str
    
class MessageResponse(BaseModel):
    response: str
    debug_info: Optional[Dict[Any, Any]] = None

class HealthResponse(BaseModel):
    status: str
    assistant_id: str
    timestamp: float

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    try:
        # Verify assistant exists
        assistant = client.beta.assistants.retrieve(ASSISTANT_ID)
        print(f"Assistant ID: {assistant}")
       
        
        return HealthResponse(
            status="healthy",
            assistant_id=ASSISTANT_ID,
            timestamp=time.time(),
            assistant_name=assistant.name if hasattr(assistant, 'name') else "N/A",
        )
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/chat", response_model=MessageResponse)
async def chat_with_assistant(request: MessageRequest):
    """Send message to existing OpenAI assistant"""
    
    
    debug_info = {
        "request_timestamp": time.time(),
        "assistant_id": ASSISTANT_ID,
        "message_length": len(request.message)
    }
    
    try:
        
        thread = client.beta.threads.create()
        thread_id = thread.id
        debug_info["thread_id"] = thread_id
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=request.message
        )
        
        # Run the assistant
        logger.debug(f"Running assistant {ASSISTANT_ID}")
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )
        
        debug_info["run_id"] = run.id
        debug_info["run_status"] = run.status
        
        if run.status == 'completed':

            messages = client.beta.threads.messages.list(thread_id=thread_id)
            assistant_message = messages.data[0]  # Latest message (assistant's response)
            assistant_response = assistant_message.content[0].text.value
            
            # Clean up - delete the temporary thread
            # logger.debug(f"Cleaning up thread {thread_id}")
            client.beta.threads.delete(thread_id)
            
            # debug_info["response_timestamp"] = time.time()
            # debug_info["total_time"] = debug_info["response_timestamp"] - debug_info["request_timestamp"]
            # debug_info["cleanup_completed"] = True
            # logger.debug(f"Response preview: {assistant_response[:100]}...")
            
            return MessageResponse(
                response=assistant_response,
                debug_info=debug_info
            )
        else:
            # Clean up failed thread
            try:
                client.beta.threads.delete(thread_id)
            except:
                pass
            
        
            raise HTTPException(status_code=500, detail=f"Assistant run failed: {run.status}")
        
    except Exception as e:
        # logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        # debug_info["error"] = str(e)
        # debug_info["error_timestamp"] = time.time()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/assistant/info")
async def get_assistant_info():
    """Get information about the configured assistant"""
    
    try:
        assistant = client.beta.assistants.retrieve(ASSISTANT_ID)
        
        return {
            "id": assistant.id,
            "name": assistant.name,
            "description": assistant.description,
            "model": assistant.model,
            "tools": assistant.tools if hasattr(assistant, 'tools') else [],
            "created_at": assistant.created_at
        }
        
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=f"Error retrieving assistant info: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "OpenAI Assistant API",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)",
            "assistant_info": "/assistant/info",
            "docs": "/docs"
        }
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")