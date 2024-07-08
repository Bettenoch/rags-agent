import nest_asyncio
import asyncio
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from helper import get_openai_api_key

nest_asyncio.apply()

OPENAI_API_KEY = get_openai_api_key()

async def get_router_query_engine(file_path: str, query: str, pages: List[str] = None) -> str:
    pages = pages or []
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

    # Load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    # Split documents into nodes
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # Create indexes
    vector_index = VectorStoreIndex(nodes)
    summary_index = SummaryIndex(nodes)

    # Define vector query function
    def vector_query(
        query: str, 
        page_numbers: List[str]
    ) -> str:
        """Perform a vector search over an index."""
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response

    # Create query tools
    vector_query_tool = FunctionTool.from_defaults(
        name="vector_tool",
        fn=vector_query
    )

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name="summary_tool",
        query_engine=summary_query_engine,
        description="Useful if you want to get a summary of the document"
    )

    # Perform query
    response = await llm.predict_and_call(
        [vector_query_tool, summary_tool], 
        query, 
        verbose=True
    )

    return response

# Example usage
if __name__ == "__main__":
    async def main():
        file_path = "./datasets/lora.pdf"  # Replace with your actual file path
        query = "What are LOW-RANK-PARAMETRIZED UPDATE MATRICES?"
        response = await get_router_query_engine(file_path, query)
        print(response)

    asyncio.run(main())
