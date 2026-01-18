from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI() # Create FastAPI instance
class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None   

@app.get('/')
def get_home():
    return {"message": "Welcome to the FastAPI application!"}

@app.get('/user/{user_id}')
def get_user(user_id: int):
    return{"user_id":user_id}

@app.post('/items/')
def create_item(item: Item):
    return item

@app.put('/items/{item_id}')
def update_item(item_id: int, item: Item):
    return {"item_id": item_id, "item": item}



