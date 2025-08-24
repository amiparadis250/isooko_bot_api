from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import openai
import logging
import time
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Isooko API", version="1.0.0")

# Env vars
ASSISTANT_API_KEY = os.getenv("Apikey")
ASSISTANT_ID = os.getenv("assistantId")

client = openai.OpenAI(api_key=ASSISTANT_API_KEY)

# -------------------------
# Models
# -------------------------
class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    response: str
    debug_info: Optional[Dict[Any, Any]] = None

class HealthResponse(BaseModel):
    status: str
    assistant_id: str
    timestamp: float

# -------------------------
# WebSocket Manager
# -------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected.")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected.")

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

# -------------------------
# POST Chat Endpoint (existing)
# -------------------------
@app.post("/chat", response_model=MessageResponse)
async def chat_with_assistant(request: MessageRequest):
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

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )

        debug_info["run_id"] = run.id
        debug_info["run_status"] = run.status

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            assistant_message = messages.data[0]
            assistant_response = assistant_message.content[0].text.value

            client.beta.threads.delete(thread_id)  # cleanup
            return MessageResponse(response=assistant_response, debug_info=debug_info)
        else:
            raise HTTPException(status_code=500, detail=f"Assistant run failed: {run.status}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# -------------------------
# WebSocket Chat Endpoint (streaming)
# -------------------------
@app.websocket("/ws/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        while True:
            user_message = await websocket.receive_text()
            logger.debug(f"Received from {client_id}: {user_message}")

            try:
                # Stream tokens from the model
                with client.chat.completions.create(
                    model="gpt-4o-mini",   # replace with your model
                    messages=[{"role": "user", "content": user_message}],
                    stream=True,
                ) as stream:
                    for event in stream:
                        delta = event.choices[0].delta
                        if delta and delta.content:
                            # Send token immediately
                            await manager.send_message(delta.content, client_id)

                    # Mark end of response
                    await manager.send_message("[[END_OF_MESSAGE]]", client_id)

            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                await manager.send_message(f"Error: {str(e)}", client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Connection closed: {client_id}")
    except Exception as e:
        manager.disconnect(client_id)
        logger.error(f"WebSocket error for {client_id}: {str(e)}")

# Other Endpoints
# -------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        assistant = client.beta.assistants.retrieve(ASSISTANT_ID)
        return HealthResponse(
            status="healthy",
            assistant_id=ASSISTANT_ID,
            timestamp=time.time(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/assistant/info")
async def get_assistant_info():
    try:
        assistant = client.beta.assistants.retrieve(ASSISTANT_ID)
        return {
            "id": assistant.id,
            "name": assistant.name,
            "description": assistant.description,
            "model": assistant.model,
            "tools": assistant.tools if hasattr(assistant, "tools") else [],
            "created_at": assistant.created_at
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving assistant info: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "OpenAI Assistant API",
        "endpoints": {
            "health": "/health",
            "chat_post": "/chat (POST)",
            "chat_ws": "/ws/chat/{client_id} (WebSocket)",
            "assistant_info": "/assistant/info",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
