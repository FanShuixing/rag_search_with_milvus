from openai import OpenAI

"""
向量化相关
"""


class Embedding:
    def __init__(self) -> None:
        self.embedding = OpenAI()

    def emb_text(self, text):
        return (
            self.embedding.embeddings.create(input=text, model="text-embedding-3-small")
            .data[0]
            .embedding
        )
