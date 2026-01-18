from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

class Product(BaseModel):
    id: int
    name: str
    price: float
