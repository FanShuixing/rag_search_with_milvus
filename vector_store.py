from pymilvus import MilvusClient
from glob import glob
from embedding import Embedding

"""
Milvus操作
"""


class VectorStore:
    def __init__(self) -> None:
        # 如果使用docker启动rag，则需要使用host.docker.internal,如果本地启动rag，则使用http://localhost:19530
        self.client = MilvusClient(
            uri="http://localhost:19530", db_name="default", timeout=10
        )
        # self.client = MilvusClient(
        #     uri="http://host.docker.internal:19530", db_name="default", timeout=10
        # )
        self.collection_name = "my_rag"
        self.vector = Embedding()

    def create_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self.client.create_collection(self.collection_name, dimension=1536)

    def insert_data(self):
        # 处理数据
        text_lines = []
        for file in glob("milvus_docs/en/faq/*.md", recursive=True):
            with open(file) as f:
                file_text = f.read()
            text_lines += file_text.split("# ")
        # 执行embeding
        vectors = [self.vector.emb_text(text) for text in text_lines]
        data = [
            {"id": i, "vector": vectors[i], "text": text_lines[i]}
            for i in range(len(vectors))
        ]

        # 插入数据
        return self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query, topk=5):
        if not self.client.has_collection(collection_name=self.collection_name):
            self.create_collection()
            self.insert_data()
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[self.vector.emb_text(query)],
            limit=topk,
            output_fields=["text"],
        )

        res_list = [each["entity"]["text"] for each in search_res[0]]
        return res_list


if __name__ == "__main__":
    rag = VectorStore()
    print(rag.search("milvus how to insert data"))
