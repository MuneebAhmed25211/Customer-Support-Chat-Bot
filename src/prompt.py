from langchain_core.prompts import PromptTemplate


prompt = PromptTemplate(template="""

Context: {context}

Question: {question}

Answer only from context:""",
input_variables=["context","question"])