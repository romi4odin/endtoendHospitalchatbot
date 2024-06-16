from pydantic import BaseModel

class HospitalQueryInput(BaseModel):
    test:str

class HospitalQueryOutput(BaseModel):
    input:str
    output:str
    intermediate_steps: list[str]