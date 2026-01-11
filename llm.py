from typing import List
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (PromptTemplate,
                                    ChatPromptTemplate,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

class LLM:
    """
    Summarize docs and generates answer
    """
    def __init__(self, model_type: str = "ollama", temperature: float = 0.1, api_key: str = ""):
        if model_type == 'ollama':
            self.model = ChatOllama(model="llama3.1:8b", temperature=temperature)
        elif model_type == 'openai':
            self.model = ChatOpenAI(model='gpt-3.5-turbo', api_key=api_key, temperature=temperature)

        # Create a custom prompt template
        sys_prompt_tmp = SystemMessagePromptTemplate.from_template(template="""
            Ты ассистент по саморефлексии и самоанализу.
            Ты отвечаешь на вопрос пользователя {user_name}. Для ответа используй информацию из Резюме по личности 
            пользователя и Контекст к его вопросу.
            
            Резюме по личности пользователя: {summary}
            
            Ответ должен содержать анализ и\или совет, в виде завершенной мысли без вопросов и уточннеий. 
            Не придумывай факты. 
                        
            ### Формат и содержание ответа:
            - Используй информацию только из Резюме и Контекста
            - Не придумывай факты, мысли, чувства пользователя
            - Не давай деструктивных советов
            - Используй вежливый и доброжелательный тон
            - Выражай поддержку и сочувствие
            - не будь категоричен, не ставь диагнозы  
            """)
        hum_prompt_tmp = HumanMessagePromptTemplate.from_template(template="""
            Контекст:{context}
            Вопрос пользователя:{query}""")

        self.promptTmp = ChatPromptTemplate.from_messages([sys_prompt_tmp, hum_prompt_tmp])
        self.chain = self.promptTmp | self.model | StrOutputParser()


    def summarize(self, docs: List[Document]):

        summarize_prompt_template = """Резюмируй основное содержание следующего текста в 5 пунктах: {text}"""
        summarize_prompt = PromptTemplate.from_template(template=summarize_prompt_template)
        summarization_chain = summarize_prompt | self.model | StrOutputParser()

        #summarization 1 stage
        docs_summaries = summarization_chain.batch([{"text":doc.page_content} for doc in docs if doc.page_content != ""])

        #summarization 2 stage
        summaries = []
        texts_bind_num = len(docs_summaries) // 4
        for i in range(0, len(docs_summaries), texts_bind_num):
            summaries.append('\n'.join(docs_summaries[i:i + texts_bind_num]))
        summaries = summarization_chain.batch([{"text": summary} for summary in summaries])

        #summarization final stage
        summary = summarization_chain.invoke({"text":"\n".join(summaries)})
        return summary