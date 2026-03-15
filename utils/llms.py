import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class LLMModel:
    def __init__(self, model_name="gpt-4o"):
        if not model_name:
            raise ValueError("Model is not defined.")
        self.model_name = model_name
        self.openai_model = ChatOpenAI(model=self.model_name)
        
    def get_model(self):
        return self.openai_model

if __name__ == "__main__":
    llm_instance = LLMModel()  
    llm_model = llm_instance.get_model()
    response=llm_model.invoke("hi")

    print(response)
