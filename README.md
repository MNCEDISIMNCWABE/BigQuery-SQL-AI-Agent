# BigQuery SQL AI Agent Powered by LangGraph

This repo demonstrates how to leverage LangGraph to build a conversational AI agent that helps users query BigQuery databases using natural language, with built-in validation, execution, and self-correction capabilities. This AI Agent takes natural language queries and convert them to SQL queries.
![image](https://github.com/user-attachments/assets/f7af3c99-912f-4630-a13e-205c8e5806fc)


#### Features
- Natural Language to SQL: Convert plain English questions into BigQuery SQL queries
- Schema-Aware: Understands your BigQuery database schema and relationships
- Self-Correcting: Automatically detects and fixes common SQL errors
- Human-in-the-Loop: Provides approval workflow for generated queries
- Interactive UI: Streamlit-based interface for easy interaction
- Vertex AI Integration: Powered by Google's LLM (Gemini) for SQL generation

#### Prerequisites
- Google Cloud Project with BigQuery and Vertex AI enabled
- Service account credentials with BigQuery access
- Python 3.8+ environment

#### Installation
Clone this repository:
- ``git clone https://github.com/MNCEDISIMNCWABE/BigQuery-SQL-AI-Agent.git``
- ```cd BigQuery-SQL-AI-Agent```

pip install -r requirements.txt
