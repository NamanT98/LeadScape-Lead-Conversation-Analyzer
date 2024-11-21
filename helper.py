from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage


class Chatbot:
    def __init__(self) -> None:
        self.system = SystemMessage(
            content="""
        You are a highly skilled salesperson.
        Your task is to understand the given conversation between client and salesperson and 
        modify the responses of salesperson such that the probablity of the client purchasing the product is increased. Do not modify the responses of 
        client. Also do not change any information about the product. 
        """
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.7,
            top_p=0.85,
        )

    def create_prompt(self, query):
        prompt = f"""Analyze the following text:
        
        {query}
        """

        return prompt

    def run_chain(self, query):
        response = self.llm.invoke(
            [self.system, HumanMessage(content=self.create_prompt(query))]
        )
        return response.content
