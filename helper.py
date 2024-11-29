from dotenv import load_dotenv

load_dotenv()

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from core import *
sdf

class PipelineBot:
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

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.summ_tokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
        self.summarizer = AutoModelForSeq2SeqLM.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary").to(self.device)

        self.score_tokenizer=Tokenizer()
        vocab_size = self.score_tokenizer.vocab_size()
        self.score_model = CustomTransformer(5650, 128, 4, 2, 50, 1).to(self.device)
        self.score_model.load_state_dict(torch.load('models/model_epoch_10.pth',weights_only=True))


    def string_to_conversation(self,string:str)->list:
        return string.split('\n')
    
    def get_summary(self,input_text):
        inputs = self.summ_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        summary_ids = self.summarizer.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=350,  # Adjust depending on the desired summary length
            min_length=40,   # Minimum length of the summary
            num_beams=4,     # Number of beams for beam search
            length_penalty=2.0
            
        )
        summary = self.summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def get_score(self,input_text):
        self.score_model.eval()

        encoded_example = self.score_tokenizer.encode(input_text, 50)
        input_tensor = torch.tensor([encoded_example], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output = self.score_model(input_tensor).squeeze() 
            predicted_score = output.item()

        return predicted_score


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