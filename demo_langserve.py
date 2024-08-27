from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uvicorn
from langserve import add_routes


prompt_template = ChatPromptTemplate.from_messages([
    "system","translate the following into {language}",
    "user","{text}"]
)

model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.7, convert_system_message_to_human=True)

parser = StrOutputParser()


chain = prompt_template | model | parser

app = FastAPI(
    title= "LLM API",
    description="My first LLM API",
    version="1.0",
)



add_routes(
    app,
    chain,
    path="/chain"
)



if __name__=="__main__":
    uvicorn.run(app, host="localhost",port=8000)