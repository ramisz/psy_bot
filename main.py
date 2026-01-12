import listener
import db_manager
import llm
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os
import streamlit as st


def main():
    st.title("Ассистент психологической консультации")

    #================ INITIALIZING =========================
    user_name = st.sidebar.text_input("Имя пользователя", value="пользователь")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.radio("Выбор ИИ провайдера", ["Ollama", "OpenAI"]) == "Ollama":
        lm = llm.LLM(temperature=0.5, model_type = 'ollama')
    else:
        api_key = st.sidebar.text_input("API KEY")
        lm = llm.LLM(temperature=0.5, model_type = "openai", api_key=api_key)

    user_summary = ""
    if os.path.isfile("summary.txt"):
        with open("summary.txt", "r") as f:
            user_summary = f.read()

    db = db_manager.DBManager()
    retriever = db.db.as_retriever(search_type='similarity', search_kwargs={'k':3})

    docs = []

    #================ ASR =========================
    uploaded_files = st.sidebar.file_uploader("Выберите аудио файл",
                                      type=["wav", "mp3", "m4a", "flac"],
                                      accept_multiple_files=True)

    if st.sidebar.button("Распознать") and (uploaded_files is not None):
        recognizer = listener.Listener()
        st.sidebar.write("Распознавание ... подождите")
        prog_bar = st.sidebar.progress(0)

        for i, file in enumerate(uploaded_files, 1):
            text = recognizer.recognize(file.name)
            prog_bar.progress(i/len(uploaded_files))
            docs.append(Document(page_content=text, metadata={"source": file.name}))
            with open(f"{i}-{file.name}.txt", 'w') as f:
                f.write(text)

    #================ DB fitting =========================
    if st.sidebar.button("Загрузить в БД"):
        res = db.append_files()
        if res == 1:
            st.sidebar.info("Файлы загружены в БД")
        elif res == 0:
            st.sidebar.error("Нет файлов для загрузки")
        elif res == -1:
            st.sidebar.error("Ошибка загрузки")

        #summarization
        chunks = db.split_data(docs=db.load_data(), chunk_size=5000, chunk_overlap=0)
        user_summary = lm.summarize(docs=[Document(page_content=user_summary)] + chunks)
        if user_summary != "":
            with open("summary.txt", 'w') as f:
                f.write(user_summary)

    if st.sidebar.button("Резюме личности"):
        with st.chat_message("assistant"):
            st.markdown(user_summary)

    #================ rendering chat ================
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #================ user input handler ================
    if question := st.chat_input("Введите запрос"):
        st.session_state.messages.append({"role":"user", "content":question})
        with st.chat_message("user"):
            st.markdown(question)

    #================ get response from the model ================
        with (st.chat_message("assistant")):
            rag = {"context":retriever | RunnableLambda(lambda docs: "\n".join([doc.page_content for doc in docs])),
                   "query": RunnablePassthrough(),
                   "user_name": RunnableLambda(lambda _: user_name),
                   "summary": RunnableLambda(lambda _: user_summary)} | lm.chain
            #print(retriever.invoke(question))
            response = rag.invoke(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

if __name__ == "__main__":

     main()
