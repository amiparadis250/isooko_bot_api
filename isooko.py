from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "welcome to Isooko chatbot API"}
