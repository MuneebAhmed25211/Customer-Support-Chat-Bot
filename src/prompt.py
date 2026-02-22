from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

memory_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support assistant.
Use the following context to answer the question.
Answer only from the context provided.
If you don't know, say you don't know.

Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])