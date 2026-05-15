"""
Spark RAG Agent

A LangGraph-based agent that integrates Spark DataFrame analytics with RAG (Retrieval-Augmented Generation)
for querying both structured sales data and unstructured corporate knowledge base.

Features:
- Structured data querying via Spark DataFrame agent
- Unstructured document retrieval via vector store
- Orchestrated workflow using LangGraph
- Exception handling for robust operation
"""

import logging
import os
import sys
from typing import Annotated, Sequence, TypedDict

import pyspark
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.agents import create_spark_dataframe_agent
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SparkRAGAgent:
    """Main agent class for Spark RAG operations."""

    def __init__(self):
        """Initialize the agent with all components."""
        try:
            self._setup_environment()
            self._initialize_spark()
            self._initialize_llms()
            self._setup_knowledge_base()
            self._setup_spark_tools()
            self._build_graph()
            logger.info("Spark RAG Agent initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    def _setup_environment(self):
        """Set up environment variables for PySpark and OpenAI."""
        try:
            # Force Worker & Driver Path Alignment
            venv_python = sys.executable
            os.environ["PYSPARK_PYTHON"] = venv_python
            os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python

            # Set OpenAI API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            os.environ["OPENAI_API_KEY"] = api_key
            logger.info("Environment setup completed.")
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            raise

    def _initialize_spark(self):
        """Initialize Spark session."""
        try:
            from pyspark.sql import SparkSession

            self.spark = (
                SparkSession.builder.appName("SparkRAGAgent")
                .config("spark.driver.memory", "4g")
                .getOrCreate()
            )
            logger.info("Spark session initialized.")
        except Exception as e:
            logger.error(f"Spark initialization failed: {e}")
            raise

    def _initialize_llms(self):
        """Initialize LLM instances."""
        try:
            # Orchestrator LLM handles top-level user routing decisions
            self.orchestrator_llm = ChatOpenAI(model="gpt-4o", temperature=0)

            # Isolated Agent LLM prevents message state bleeding from the Spark Agent loop
            self.spark_internal_llm = ChatOpenAI(model="gpt-4o", temperature=0)
            logger.info("LLMs initialized.")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise

    def _setup_knowledge_base(self):
        """Set up the unstructured knowledge base with vector store."""
        try:
            # Read documents from file
            documents_file = "documents.txt"
            if not os.path.exists(documents_file):
                raise FileNotFoundError(
                    f"Documents file '{documents_file}' not found. Please create it with your knowledge base content."
                )

            with open(documents_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Split content into individual documents (assuming each document starts with "Document ID")
            sample_unstructured_docs = [
                doc.strip() for doc in content.split("\n\n") if doc.strip()
            ]

            if not sample_unstructured_docs:
                raise ValueError(
                    f"No documents found in '{documents_file}'. Please ensure the file contains properly formatted documents."
                )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200, chunk_overlap=20
            )
            split_docs = text_splitter.create_documents(sample_unstructured_docs)

            embeddings = OpenAIEmbeddings()
            # Create Chroma vector store with persistence
            persist_directory = "./chroma_db"
            vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
            # Persist the vector store to disk
            vector_store.persist()
            self.retriever = vector_store.as_retriever(search_kwargs={"k": 1})
            logger.info(
                f"Knowledge base setup completed with {len(sample_unstructured_docs)} documents from file."
            )
        except Exception as e:
            logger.error(f"Knowledge base setup failed: {e}")
            raise

    def _setup_spark_tools(self):
        """Set up Spark DataFrame and related tools."""
        try:
            sales_data = [
                ("Alice", "Electronics", 1200, "2026-01-15"),
                ("Bob", "Software", 450, "2026-02-11"),
                ("Charlie", "Electronics", 2900, "2026-03-01"),
                ("Alice", "Software", 150, "2026-04-10"),
            ]
            schema = ["Employee", "Department", "Revenue", "TransactionDate"]
            self.sales_df = self.spark.createDataFrame(sales_data, schema)

            # Construct internal dataframe worker with its own isolated model instance
            self.spark_agent_executor = create_spark_dataframe_agent(
                llm=self.spark_internal_llm,
                df=self.sales_df,
                verbose=False,
                allow_dangerous_code=True,
            )
            logger.info("Spark tools setup completed.")
        except Exception as e:
            logger.error(f"Spark tools setup failed: {e}")
            raise

    def _build_graph(self):
        """Build the LangGraph workflow."""
        try:
            # Define tools
            @tool
            def query_knowledge_base(query: str) -> str:
                """Searches corporate policy and unstructured knowledge documentation."""
                try:
                    docs = self.retriever.invoke(query)
                    return "\n\n".join([str(d.page_content) for d in docs])
                except Exception as e:
                    logger.error(f"Knowledge base query failed: {e}")
                    return "Error querying knowledge base."

            @tool
            def query_sales_records(query: str) -> str:
                """Queries transactional sales databases containing Revenue, Employees, and Departments."""
                try:
                    response = self.spark_agent_executor.invoke({"input": query})
                    if isinstance(response, dict) and "output" in response:
                        return str(response["output"])
                    return str(response)
                except Exception as e:
                    logger.error(f"Sales records query failed: {e}")
                    return "Error querying sales records."

            # State definition
            class AgentState(TypedDict):
                messages: Annotated[Sequence[BaseMessage], add_messages]

            self.tools_list = [query_knowledge_base, query_sales_records]
            tool_node = ToolNode(self.tools_list)

            # Bind the tools list to the top-level orchestrator model instance
            model_with_tools = self.orchestrator_llm.bind_tools(self.tools_list)

            def call_model(state: AgentState):
                try:
                    response = model_with_tools.invoke(state["messages"])
                    return {"messages": [response]}
                except Exception as e:
                    logger.error(f"Model call failed: {e}")
                    return {
                        "messages": [HumanMessage(content="Error processing request.")]
                    }

            def should_continue(state: AgentState):
                try:
                    last_message = state["messages"][-1]
                    if not last_message.tool_calls:
                        return END
                    return "tools"
                except Exception as e:
                    logger.error(f"Continue check failed: {e}")
                    return END

            # Assemble Execution Graph
            workflow = StateGraph(AgentState)
            workflow.add_node("agent", call_model)
            workflow.add_node("tools", tool_node)

            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", should_continue)
            workflow.add_edge("tools", "agent")

            self.app = workflow.compile()
            logger.info("Graph built successfully.")
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            raise

    def run_query(self, query: str) -> str:
        """Run a query through the agent and return the response."""
        try:
            result = self.app.invoke({"messages": [HumanMessage(content=query)]})
            return result["messages"][-1].content
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return "An error occurred while processing the query."

    def stream_query(self, query: str):
        """Stream the query execution for debugging."""
        try:
            state_input = {"messages": [HumanMessage(content=query)]}
            for output in self.app.stream(state_input, stream_mode="updates"):
                for node_name, node_state in output.items():
                    print(f"Node '{node_name}' successfully processed state.")
                    if node_name == "agent":
                        latest_msg = node_state["messages"][-1]
                        if latest_msg.content:
                            print(f"Agent Text: {latest_msg.content}")
        except Exception as e:
            logger.error(f"Stream query failed: {e}")
            print("An error occurred during streaming.")


def main():
    """Main execution function."""
    try:
        agent = SparkRAGAgent()

        print("\n=== Test 1: Querying Unstructured Vector Data ===")
        query_1 = "What is the policy regarding travel expenses timeline?"
        agent.stream_query(query_1)

        print("\n=== Test 2: Querying Structured Spark DataFrame ===")
        query_2 = "Calculate the total revenue generated specifically by Alice across all departments."
        result = agent.run_query(query_2)
        print(f"Final Integrated Answer:\n{result}")

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print("Failed to run the agent.")


if __name__ == "__main__":
    main()
