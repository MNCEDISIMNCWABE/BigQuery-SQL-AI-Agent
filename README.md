# BigQuery SQL AI Agent Powered by LangGraph

This repo demonstrates how to leverage LangGraph to build a conversational AI agent that helps users query BigQuery databases using natural language, with SQL validation, execution, and self-correction capabilities. This AI Agent takes natural language queries and convert them to SQL queries.

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
- Install dependencies: ```pip install -r requirements.txt```

#### Set up your Google Cloud credentials
- Place the service account in the project directory: ```os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "path/to/your/credentials.json"```

#### Other Key Configurations
```
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001", 
    temperature=0,                     
    max_output_tokens=2048,             
    project=BQ_PROJECT,
    location=LOCATION
)

BQ_PROJECT = "your-project-id"         
BQ_DATASET = "your-dataset-name"       
LOCATION = "your-bq-region"
```

#### Run it
``streamlit run app.py``

Once launched, simply enter the your query like:

<img width="996" alt="image" src="https://github.com/user-attachments/assets/19995b63-20d5-4099-906a-fbe95791664b" />



