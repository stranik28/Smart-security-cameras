from fastapi import FastAPI

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
import json

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stats")
async def get_stats():
    dictg = {
        'count': 122,
        '46': 12,
        '812': 21,
        '1520': 12,
        '2532':43,
        '3843':9,
        '4853':12,
        '60100':32,
        'male': 50,
        'female':'50'
    }
    return json.dumps(dictg)
    