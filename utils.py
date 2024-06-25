from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.tools import FunctionTool
from typing import List




def get_router_query_engine(file_path: str, llm = None, embed_model = None):
    """Get router query engine."""
    llm = llm or OpenAI(model="gpt-3.5-turbo")
    embed_model = embed_model or OpenAIEmbedding(model="text-embedding-ada-002")

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    
    def vector_query_engine(
        query: str,
        page_numbers: List[int]
    ) -> str:
        """ Perform a vector search over an index.
        query (str): the string query to be embedded.
        page_numbers (List[int]): Filter by set of pages. Leave blank id we want to perform a vector  search 
        over all pages. otherwise, filter by the set of specified pages
        
        """
        
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            ),
        )
        response = query_engine.query(query)
        return response
    
    vector_query_tool = FunctionTool.from_defaults(
        name="vector_query_tool",
        fn=vector_query_engine
    )

    summary_tool = QueryEngineTool.from_defaults(
        name="summary_tool",
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to MetaGPT"
        ),
    )

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    
    response = llm.predict_and_call(
        [vector_query_tool, summary_tool],
        input="what are Subspace similarity between different random seeds as described in page 11?",
        stop=["\n\n"],
        verbose=True
    )

    return response

