import streamlit as st
from langchain_google_community import BigQueryLoader
from google.cloud import bigquery
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import Graph, END
from typing import TypedDict, Annotated, Union, Optional
import pandas as pd
import json
import os
import re
import warnings
warnings.filterwarnings("ignore")

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/path/to/google_service_account.json"

# Initialize BigQuery client
client = bigquery.Client()

# Configuration
BQ_PROJECT = "<your-bq-project" 
BQ_DATASET = "<your-bq-dataset>"   
LOCATION = "<your-bq-region>"  

# Initialize Vertex AI LLM
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001", # Can change to other Vertex AI LLM models if needed
    temperature=0,                     # Controls creativity (0 for deterministic)
    max_output_tokens=2048,            # Maximum SQL query length
    project=BQ_PROJECT,
    location=LOCATION
)

def get_all_schemas(dataset_id):
    """Fetch schemas for all tables in the dataset"""
    tables = client.list_tables(dataset_id)
    schemas = {}
    
    for table in tables:
        try:
            table_ref = client.get_table(table)
            schemas[table.table_id] = [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "description": field.description or "No description"
                }
                for field in table_ref.schema
            ]
        except Exception as e:
            print(f"Error fetching schema for {table.table_id}: {str(e)}")
    
    return schemas

def format_schema_for_prompt(schema_data):
    """Format schema data for human-readable prompt"""
    formatted = []
    for table_name, columns in schema_data.items():
        formatted.append(f"Table {table_name}:")
        for col in columns:
            formatted.append(f"  - {col['name']} ({col['type']}): {col['description']}")
    return "\n".join(formatted)

def get_table_reference(table_name: str) -> str:
    """Generate properly formatted BigQuery table reference"""
    return f"`{BQ_PROJECT}.{BQ_DATASET}.{table_name}`"

# Get all schemas
schemas = get_all_schemas(BQ_DATASET)

# agent instructions prompt - YOU CAN MODIFY THIS TO SUITE YOUR BQ TABLES
system_prompt = f"""You are an advanced BigQuery SQL expert with data modeling intuition. Key capabilities:

1. Schema Reasoning:
- Automatically detect date fields that should join to dim_date (e.g., first_seen_date → dim_date.date)
- Recognize common patterns (user_id for joins, *_date for time dimensions)
- Identify fact vs dimension tables based on structure

2. Intelligent Defaults:
- For time-based questions, default to appropriate date granularity (month/quarter/year)
- For user metrics, consider both raw (user-engagement) and aggregated (fact_user_metrics) sources
- When counting distinct values, automatically add LIMIT based on expected cardinality

3. Self-Correction:
- If initial query returns unexpected zeros/null values:
  1. Check date formatting
  2. Verify join conditions
  3. Consider alternative source tables

4. Analytical Best Practices:
- Prefer COUNT(DISTINCT) over COUNT() for user metrics
- Use appropriate date functions (EXTRACT, DATE_TRUNC)
- Apply CASE WHEN for conditional logic

Available tables:
{format_schema_for_prompt(schemas)}

Examples of Intelligent Behavior:
Q: "How many unique months per user?"
A: SELECT 
     u.user_id,
     COUNT(DISTINCT FORMAT_DATE('%Y-%m', d.date)) AS unique_months
   FROM {get_table_reference('user-engagement')} u
   JOIN {get_table_reference('dim_date')} d 
     ON u.first_seen_date = d.date
   GROUP BY u.user_id

Q: "Find users active in Q2 but not Q3"
A: WITH q2_users AS (
     SELECT DISTINCT user_id 
     FROM {get_table_reference('user-engagement')} u
     JOIN {get_table_reference('dim_date')} d 
       ON u.first_seen_date = d.date
     WHERE d.quarter = 'Q2'
   )
   SELECT q2.user_id
   FROM q2_users q2
   WHERE NOT EXISTS (
     SELECT 1 
     FROM {get_table_reference('user-engagement')} u2
     JOIN {get_table_reference('dim_date')} d2 
       ON u2.first_seen_date = d2.date
     WHERE d2.quarter = 'Q3'
     AND u2.user_id = q2.user_id
   )
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# Define the agent state
class AgentState(TypedDict):
    question: str
    sql_query: Optional[str]
    query_result: Union[pd.DataFrame, str, None]
    validation_errors: list[str]
    attempts: int
    needs_correction: bool
    human_approved: Optional[bool]
    human_feedback: Optional[str]
    needs_modification: Optional[bool]

def clean_sql_query(query: str) -> str:
    """Clean and normalize SQL query by removing backticks, etc."""
    if query is None:
        return ""
    if query.startswith("```") and query.endswith("```"):
        query = query.strip("`")
        if query.lower().startswith("sql"):
            query = query[3:].strip()
    return query.strip()

def generate_sql(state: AgentState) -> AgentState:
    """Generate SQL query from question"""
    chain = (
        {"question": lambda x: x["question"]}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    sql_query = chain.invoke(state)
    return {
        **state,
        "sql_query": sql_query, 
        "attempts": state.get("attempts", 0) + 1,
        "query_result": None, 
        "validation_errors": [] 
    }

def validate_sql(state: AgentState) -> AgentState:
    """Validate SQL query syntax"""
    query = clean_sql_query(state["sql_query"])
    errors = []
    
    table_ref_pattern = re.compile(
        r"(`" + re.escape(BQ_PROJECT) + r"\." + re.escape(BQ_DATASET) + r"\.[a-zA-Z0-9_-]+`|`[a-zA-Z0-9_-]+`)", 
        re.IGNORECASE
    )
    
    # Validation checks
    if not query.lower().startswith(("select", "with")):
        errors.append("Must start with SELECT/WITH")
    
    if not table_ref_pattern.search(query):
        errors.append(f"Missing valid table reference (expected format: `{BQ_PROJECT}.{BQ_DATASET}.table_name` or `table_name`)")
    
    if "join" in query.lower() and not re.search(r"\bjoin\b(.|\n)+?\bon\b", query, re.IGNORECASE):
        errors.append("JOIN missing ON clause")
    
    return {
        **state,  
        "validation_errors": errors, 
        "sql_query": state["sql_query"],
        "query_result": None  
    }

def execute_query(state: AgentState) -> AgentState:
    """Execute the SQL query in BigQuery"""
    if state["validation_errors"]:
        return {
            **state,  
            "query_result": f"Validation errors: {', '.join(state['validation_errors'])}",
            "sql_query": state["sql_query"],
            "needs_correction": True
        }
    
    query = clean_sql_query(state["sql_query"])
    try:
        query_job = client.query(query)
        result = query_job.result().to_dataframe()
        return {
            **state, 
            "query_result": result,
            "sql_query": state["sql_query"],
            "needs_correction": False
        }
    except Exception as e:
        return {
            **state,  
            "query_result": f"Execution Error: {str(e)}",
            "sql_query": state["sql_query"],
            "needs_correction": True
        }

def analyze_results(state: AgentState) -> AgentState:
    """Analyze query results for potential issues"""
    result_update = {
        **state, 
        "sql_query": state["sql_query"],
        "query_result": state["query_result"]
    }
    
    if isinstance(state["query_result"], str):
        result_update["needs_correction"] = True
        return result_update
    
    if isinstance(state["query_result"], pd.DataFrame):
        df = state["query_result"]
        # Handle empty results
        if df.empty:
            result_update["needs_correction"] = True
            result_update["query_result"] = "Query returned empty results"
            return result_update
            
        if len(df.columns) == 1:
            if df.iloc[:, 0].eq(0).all():
                result_update["needs_correction"] = True
                result_update["query_result"] = "Query returned all zeros"
        else:
            if (df.iloc[:, 1:] == 0).all().all():
                result_update["needs_correction"] = True
                result_update["query_result"] = "Query returned all zeros"
            else:
                result_update["needs_correction"] = False
    
    return result_update

def correct_query(state: AgentState) -> AgentState:
    """Correct the SQL query based on errors"""
    error_context = state.get("query_result", "Unknown error")
    
    if state.get("human_approved") is False:
        error_context = "Human reviewer rejected the generated SQL query"
    
    original_query = state.get("sql_query", "No query generated yet")
    
    question = state.get("question", "Unknown question") 
    
    correction_prompt = f"""Correct this SQL query based on the feedback:
    
    Error Context: {error_context}
    Original Query: {original_query}
    
    User Question: {question}
    
    Provide ONLY the corrected SQL query:"""
    
    corrected = llm.invoke(correction_prompt)
    return {
        **state,  
        "sql_query": corrected.content.strip(),
        "attempts": state.get("attempts", 0),
        "query_result": None,
        "validation_errors": [],
        "human_approved": None
    }

def modify_query_based_on_feedback(state: AgentState) -> AgentState:
    """Modify query based on human feedback"""
    original_query = state["sql_query"]
    feedback = state.get("human_feedback", "")
    
    modification_prompt = f"""
    I have a SQL query that needs to be modified based on human feedback.
    
    Original SQL query:
    {original_query}
    
    Human feedback: {feedback}
    
    Please modify the query according to this feedback. Return ONLY the modified SQL query.
    """
    
    modified_query = llm.invoke(modification_prompt)
    
    return {
        **state,
        "sql_query": modified_query.content.strip(),
        "needs_modification": False,
        "human_approved": True
    }

def initialize_workflow():
    """Initialize and configure the workflow graph"""
    workflow = Graph()

    # Add nodes
    workflow.add_node("generate", generate_sql)
    workflow.add_node("validate", validate_sql)
    workflow.add_node("execute", execute_query)
    workflow.add_node("analyze", analyze_results)
    workflow.add_node("correct", correct_query)
    workflow.add_node("modify", modify_query_based_on_feedback)

    # Set up workflow edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "validate")
    
    # After validation, show to human for approval
    workflow.add_conditional_edges(
        "validate",
        lambda x: "execute" if not x.get("validation_errors") else "correct",
        {"execute": "execute", "correct": "correct"}
    )
    
    workflow.add_edge("execute", "analyze")

    workflow.add_conditional_edges(
        "analyze",
        lambda x: "correct" if x.get("needs_correction", False) and x.get("attempts", 0) < 3 else END,
        {"correct": "correct", END: END}
    )

    workflow.add_edge("correct", "validate")
    
    return workflow.compile()

# Initialize the workflow
app = initialize_workflow()

def bigquery_agent(question: str, max_attempts: int = 3) -> Union[pd.DataFrame, str]:
    """Run the agent workflow"""
    state = {
        "question": question,
        "sql_query": None,
        "query_result": None,
        "validation_errors": [],
        "attempts": 0,
        "needs_correction": False,
        "human_approved": None,
        "human_feedback": None,
        "needs_modification": False
    }
    
    for attempt in range(max_attempts):
        result_state = app.invoke(state, {"recursion_limit": 50})
        
        if result_state is None:
            break
        else:
            state = result_state
        
        if not state.get("needs_correction", False):
            result = state.get("query_result")
            if isinstance(result, pd.DataFrame):
                return result
            else:
                return f"Final result: {result}"
    
    return state.get("query_result", "Max attempts reached without success")


def main():
    st.title("BigQuery SQL Agent")
    st.write("Ask questions about your BigQuery data and get SQL answers")
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'feedback' not in st.session_state:
        st.session_state.feedback = ""
    if 'agent_state' not in st.session_state:
        st.session_state.agent_state = None
    
    # Question input
    question = st.text_input("Enter your question about the data:", key="question_input")
    
    if st.button("Submit Question"):
        if question:
            st.session_state.current_question = question
            st.session_state.agent_state = {
                "question": question,
                "sql_query": None,
                "query_result": None,
                "validation_errors": [],
                "attempts": 0,
                "needs_correction": False,
                "human_approved": None,
                "human_feedback": None,
                "needs_modification": False
            }
            
            # Run initial generation
            st.session_state.agent_state = generate_sql(st.session_state.agent_state)
            st.session_state.agent_state = validate_sql(st.session_state.agent_state)
            
            # Show SQL for approval
            st.session_state.history.append({
                "type": "question",
                "content": question
            })
            st.session_state.history.append({
                "type": "sql",
                "content": st.session_state.agent_state["sql_query"],
                "validation_errors": st.session_state.agent_state["validation_errors"]
            })
            st.rerun()
    
    # Display history
    st.subheader("Conversation History")
    for item in st.session_state.history:
        if item["type"] == "question":
            st.markdown(f"**You:** {item['content']}")
        elif item["type"] == "sql":
            st.markdown("**Generated SQL:**")
            st.code(item["content"], language="sql")
            if item.get("validation_errors"):
                st.error("Validation errors: " + ", ".join(item["validation_errors"]))
        elif item["type"] == "result":
            if isinstance(item["content"], pd.DataFrame):
                st.dataframe(item["content"])
            else:
                st.write(item["content"])
        elif item["type"] == "feedback":
            st.info(f"**Feedback:** {item['content']}")
    
    # Feedback and approval system
    if st.session_state.agent_state and st.session_state.agent_state.get("sql_query"):
        st.subheader("Query Approval")
        
        feedback = st.text_area("Provide feedback or modifications for the SQL query:", 
                              key="feedback_input",
                              help="Type 'Y' to approve, 'N' to reject, or provide specific feedback like 'Yes but order by date'")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Approve Query"):
                if feedback.lower().startswith(('y', 'yes')):
                    if 'but' in feedback.lower():
                        st.session_state.agent_state["human_approved"] = True
                        st.session_state.agent_state["human_feedback"] = feedback[feedback.lower().find('but') + 3:].strip()
                        st.session_state.agent_state["needs_modification"] = True
                    else:
                        st.session_state.agent_state["human_approved"] = True
                        st.session_state.agent_state["needs_modification"] = False
                    
                    # Execute or modify
                    if st.session_state.agent_state["needs_modification"]:
                        st.session_state.agent_state = modify_query_based_on_feedback(st.session_state.agent_state)
                        st.session_state.history.append({
                            "type": "sql",
                            "content": st.session_state.agent_state["sql_query"],
                            "validation_errors": []
                        })
                    
                    st.session_state.agent_state = execute_query(st.session_state.agent_state)
                    st.session_state.agent_state = analyze_results(st.session_state.agent_state)
                    
                    st.session_state.history.append({
                        "type": "result",
                        "content": st.session_state.agent_state["query_result"]
                    })
                    
                    st.rerun()
        
        with col2:
            if st.button("Reject Query"):
                st.session_state.agent_state["human_approved"] = False
                st.session_state.agent_state["human_feedback"] = feedback if feedback else "Query rejected"
                
                st.session_state.history.append({
                    "type": "feedback",
                    "content": st.session_state.agent_state["human_feedback"]
                })
                
                # Attempt correction
                if st.session_state.agent_state["attempts"] < 3:
                    st.session_state.agent_state = correct_query(st.session_state.agent_state)
                    st.session_state.agent_state = validate_sql(st.session_state.agent_state)
                    
                    st.session_state.history.append({
                        "type": "sql",
                        "content": st.session_state.agent_state["sql_query"],
                        "validation_errors": st.session_state.agent_state["validation_errors"]
                    })
                    
                    st.rerun()
                else:
                    st.session_state.history.append({
                        "type": "result",
                        "content": "Max attempts reached without success"
                    })
                    st.rerun()
        
        with col3:
            if st.button("Reset Session"):
                st.session_state.history = []
                st.session_state.current_question = ""
                st.session_state.feedback = ""
                st.session_state.agent_state = None
                st.rerun()

if __name__ == "__main__":
    main()
