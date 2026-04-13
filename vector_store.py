from pymilvus import MilvusClient
from glob import glob
from embedding import Embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

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
        self.embeding = Embedding()
        self.spliter = RecursiveCharacterTextSplitter(
            separators=[
                "\n## ",  # H2 标题（二级）
                "\n### ",  # H3 标题（三级）
                "\n#### ",  # H4 标题（四级）
                "\n##### ",  # H5 标题（五级）
                "\n**",  # 粗体/强调
                "\n\n",  # 段落分隔
                "\n",  # 换行
                " ",  # 单词
                "",  # 字符
            ],
            chunk_size=2000,
            chunk_overlap=200,
        )

    def create_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self.client.create_collection(self.collection_name, dimension=1536)

    def read_data(self):
        text_lines = []
        for file in glob("milvus_docs/en/**/*.md", recursive=True):
            with open(file) as f:
                file_text = f.read()
            file_text_list = self.spliter.split_text(file_text)
            text_lines += file_text_list
        return text_lines

    def emb_batch(self, texts: list, batch_size: int = 100):
        all_vectors = []
        all_texts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                res = self.embeding.emb_text(batch)
                all_vectors.extend(res)
                all_texts.extend(batch)
            except Exception as e:
                print(f"batch失败{i}:{i+batch_size},原因{e}")
        return all_vectors, all_texts

    def insert_data(self):
        # 处理数据
        print("处理数据中...")
        text_lines = self.read_data()
        print(f"一共有{len(text_lines)}个片段")

        vectors, texts = self.emb_batch(text_lines)
        print(f"vectors:{len(vectors)},texts:{len(texts)}")

        data = [
            {"id": i, "vector": vectors[i], "text": texts[i]}
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
            data=self.embeding.emb_text(query),
            limit=topk,
            output_fields=["text"],
        )

        res_list = [each["entity"]["text"] for each in search_res[0]]
        return res_list


if __name__ == "__main__":
    rag = VectorStore()
    print(rag.search("milvus how to insert data"))
