---
layout: post
title: Building an AI SQL Agent On A Live Cloud-Based Database
image: "/posts/gen-ai-sql-agent-title-img.png"
tags: [GenAI, SQL, Agents, Python, LangChain, PostgreSQL]
---

In this project we build a practical **SQL AI Agent** for our grocery retail client, capable of taking natural-language questions and turning them into PostgreSQL queries against their database.

The agent is able to; interpret the user's intent, plan how to answer it using SQL, write an appropriate query, execute that query on the database, and return a clear natural-language answer.

We achieve this by combining:

* A secure database connection
* A carefully scoped database wrapper
* A purpose-built SQL system prompt
* A set of SQL-aware tools
* A modern LLM configured as an agent

We then test the agent on two questions, and check every answer by running our own SQL against the database.

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. SQL Agent Overview](#agent-overview)
- [03. Setting Up the Database Connection](#db-setup)
    - [Environment Variables](#db-env)
    - [Postgres Connection String](#db-uri)
    - [Engine and Health Check](#db-engine)
    - [SQLDatabase Wrapper](#db-sqldatabase)
- [04. Building the SQL Agent](#agent-build)
    - [LLM Setup](#agent-llm)
    - [SQL Toolkit and Tools](#agent-toolkit)
    - [System Prompt](#agent-prompt)
    - [Creating the Agent](#agent-create)
- [05. Application & Examples](#agent-application)
- [06. Growth & Next Steps](#growth-next-steps)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

ABC Grocery holds rich customer and transaction data in a PostgreSQL database. Data Science and Analytics teams often ask questions such as:

* *Which customers live furthest from the store on average?*
* *What is the average transaction value over a given period?*
* *How do spending patterns differ by gender or credit score?*

Answering questions like these typically requires writing SQL queries by hand, which can be a bottleneck for non-technical stakeholders.

The goal of this project is to build an **AI SQL Agent** that can take a plain-English question and:

1. Understand what is being asked
2. Design a query using good SQL practices
3. Execute it safely against the database
4. Return a concise natural-language summary

### Actions <a name="overview-actions"></a>

We built an end-to-end SQL Agent that:

* Securely connects to the PostgreSQL database, with credentials kept in a *.env* file
* Exposes only the required schema and tables to the agent
* Uses a modern LLM (gpt-4.1) configured with a dedicated SQL system prompt
* Uses LangChain’s SQL tooling to inspect schemas and run queries
* Returns a human-readable answer

We then asked the agent two questions, inspected every step it took, and verified each answer by writing our own SQL and running it on the database.

### Results <a name="overview-results"></a>

| Question | Agent's answer | Our check |
|---|---|---|
| On average, which gender lives furthest from the store? | Female, 3.98 miles | ✅ Correct |
| What is the average transaction value in September 2020 for male customers with a credit score above 0.5? | $88.49 | ✅ Correct |

The agent:

* Correctly answered a question on one table, and a harder question that needed both tables
* Handled grouping, aggregation and joins
* Avoided double counting, by adding up each transaction's rows before averaging

We also found a limitation. The second question can be read in two ways, which give $88.49 and $101.32. The agent chose one reading without mentioning the other, even though the system prompt asks it to check when a question is ambiguous.

### Growth/Next Steps <a name="overview-growth"></a>

Potential future enhancements include:

* Exposing additional tables (for example, product metadata, campaign data)
* Adding a lightweight UI so non-technical users can chat with the agent
* Logging queries and responses for audit and learning
* Adding evaluation harnesses to automatically check query correctness
* Adding clarification loops when questions are ambiguous

___

# 01. Data Overview <a name="data-overview"></a>

For this project, the SQL Agent interacts with two core tables in the *grocery_db* schema:

1. **grocery_db.customer_details** – one row per customer
2. **grocery_db.transactions** – one row per combination of customer, transaction, and product area

## grocery_db.customer_details

This table stores customer-level attributes:

* *customer_id* – Unique customer identifier
* *distance_from_store* – Distance in miles from the store
* *gender* – M, F, or NULL
* *credit_score* – Decimal value between 0.00 and 1.00

Sample rows:

| **customer_id** | **distance_from_store** | **gender** | **credit_score** |
|---|---|---|---|
| 74 | 3.38 | F | 0.59 |
| 524 | 4.76 | F | 0.52 |
| 607 | 4.45 | F | 0.49 |
| 343 | 0.91 | M | 0.54 |
| 322 | 3.02 | F | 0.63 |

<br>
## grocery_db.transactions

This table is at the *customer_id, transaction_id, product_area_id* level, meaning:

* A single transaction can have multiple rows (one per product area)
* Aggregations must be done carefully to avoid double counting

Key columns:

* *customer_id* – Link back to the customer
* *transaction_date* – Date of the transaction
* *transaction_id* – Unique identifier per transaction
* *product_area_id* – Department the purchase belongs to (1–5)
* *num_items* – Number of items in that product area
* *sales_cost* – Value of those items

Sample rows:

| **customer_id** | **transaction_date** | **transaction_id** | **product_area_id** | **num_items** | **sales_cost** |
|---|---|---|---|---|---|
| 1 | 2020-04-10 | 435657533999 | 3 | 7 | 19.16 |
| 1 | 2020-04-10 | 435657533999 | 2 | 5 | 7.71 |
| 1 | 2020-06-02 | 436189770685 | 4 | 4 | 26.97 |
| 1 | 2020-06-02 | 436189770685 | 1 | 2 | 38.52 |
| 1 | 2020-06-10 | 436265380298 | 4 | 4 | 22.13 |

<br>
The first two rows show why aggregation needs care: transaction 435657533999 appears twice, once for product area 3 and once for product area 2. Counting rows would count this one shopping trip as two transactions, and its value is the sum of both rows (19.16 + 7.71 = 26.87).

The tables join via *customer_id*, and the system prompt states that the transactions table covers the period from 2020-04-01 to 2020-09-30.

___

# 02. SQL Agent Overview <a name="agent-overview"></a>

Rather than simply asking an LLM to *write some SQL*, we build a full **SQL Agent**. The difference is that a simple *SQL writer* just outputs a query (it has no direct access to the database and cannot inspect schemas or data) whereas a SQL Agent can; read the schema, inspect sample rows, choose tools to run queries, iterate based on tool results, and of course return a final answer.

In this project, the agent:

1. Receives a natural language question
2. Uses the tools provided by LangChain’s SQL toolkit to understand the schema and data
3. Generates a query that follows our design rules
4. Executes it against the database
5. Summarises the results clearly for the user, in natural language

This makes the agent both **powerful** and **constrained**, which is exactly what we want.

___

# 03. Setting Up the Database Connection <a name="db-setup"></a>

The project keeps reusable code in `src/` and uses notebooks for testing:

```
sql_agent_bot/
├── notebooks/
│   ├── 01_database_connection.ipynb
│   └── 02_sql_agent.ipynb
├── prompts/
│   └── sql-agent-system-prompt.txt
├── src/
│   ├── config.py
│   ├── database.py
│   └── agent.py
├── .env.example
├── pyproject.toml
└── uv.lock
```

<br>
Settings that aren't secret live in `src/config.py`:

```python
"""
Project settings. The notebooks and src/ files import these, so each value
only needs changing in one place. Database credentials are NOT here; they
live in .env.
"""
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# System prompt for the SQL agent (role, scope, tables, query rules, examples)
PROMPT_FILE = PROJECT_ROOT / "prompts" / "sql-agent-system-prompt.txt"

# ---------------------------------------------------------------------
# Database scope: what the agent is allowed to see
# ---------------------------------------------------------------------
SCHEMA_NAME = "grocery_db"
TABLES = ["customer_details", "transactions"]

# Number of example rows from each table shown to the agent with the schema
SAMPLE_ROWS = 5

# Any query running longer than this is cancelled (milliseconds, so 15 seconds)
STATEMENT_TIMEOUT_MS = 15000

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
LLM_MODEL = "gpt-4.1"
```

## Environment Variables <a name="db-env"></a>

We begin by loading our database and API credentials from a *.env* file, which is excluded from git:

```python
import os
from dotenv import load_dotenv
load_dotenv()
```

---

## Postgres Connection String <a name="db-uri"></a>

We construct a PostgreSQL connection string using environment variables from our .env file. It contains the database password, so it is never printed.

```python
def build_postgres_uri():
    """Build the PostgreSQL connection string from .env. Never print the result: it contains the password."""
    POSTGRES_URI = (f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
                f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DBNAME')}?sslmode=require")
    return POSTGRES_URI
```

---

## Engine and Health Check <a name="db-engine"></a>

We then create a SQLAlchemy engine and perform a quick health check:

```python
def get_engine():
    """Create the database engine and check the connection works."""
    engine = sa.create_engine(build_postgres_uri(),
                          pool_pre_ping=True,
                          connect_args={"options": f"-c statement_timeout={config.STATEMENT_TIMEOUT_MS}"} 
                          ) 

    # check the connection
    with engine.connect() as conn:
        conn.exec_driver_sql("select 1")
        

    return engine
```

Key choices:

* Pool_pre_ping = True: Validates connections before use, so a connection the cloud database closed while idle doesn't cause a failed query
* Statement_timeout = 15000: Prevents long-running queries from hanging, by cancelling any query that runs longer than 15 seconds
* The *select 1* check confirms that credentials and network access are correct

The health check matters because `create_engine` doesn't connect to anything; it only stores the settings. We tested this by putting a wrong password in *.env*: the error appeared at `engine.connect()`, not at `create_engine`.

---

## SQLDatabase Wrapper <a name="db-sqldatabase"></a>

Next, we wrap our engine in LangChain’s *SQLDatabase* utility, scoping the agent to only the tables we want:

```python
db = SQLDatabase(engine=engine,
                 schema=config.SCHEMA_NAME,
                 include_tables=config.TABLES,
                 sample_rows_in_table_info=config.SAMPLE_ROWS)

print("Usable tables:", db.get_usable_table_names())
```

```
Usable tables: ['customer_details', 'transactions']
```

Important aspects:

* schema: Explicitly set to *grocery_db*
* include_tables: Restricts access to only *customer_details* and *transactions*
* sample_rows_in_table_info = 5: Provides a small snapshot of real data to the agent

This gives the agent enough context to reason about column types and values, while keeping scope tight and safe.

We can see exactly what the agent receives about each table with `db.get_table_info()`: the table structure plus 5 sample rows (the same rows shown in the Data Overview above). For example:

```
CREATE TABLE grocery_db.transactions (
	customer_id INTEGER, 
	transaction_date DATE, 
	transaction_id BIGINT, 
	product_area_id INTEGER, 
	num_items INTEGER, 
	sales_cost NUMERIC(10, 2)
)
```

<br>
Because this comes from the live database, it also shows the real column types. For example, *transaction_id* is stored as BIGINT, while our hand-written system prompt describes it as INT.

All of the connection code lives in `src/database.py`, so the agent and notebooks can get the scoped database with one call:

```python
def get_database():
    """SQLDatabase limited to the grocery_db schema and the two tables the agent may use."""
    db = SQLDatabase(engine=get_engine(),
                 schema=config.SCHEMA_NAME,
                 include_tables=config.TABLES,
                 sample_rows_in_table_info=config.SAMPLE_ROWS)
    return db
```

___

# 04. Building the SQL Agent <a name="agent-build"></a>

## LLM Setup <a name="agent-llm"></a>

We configure a dedicated LLM for our SQL Agent:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=config.LLM_MODEL,
                 temperature=0)
```
<br>
Here we use *gpt-4.1* with a temperature of 0, which prioritises determinism, consistency, and reduced creativity (which is ideal for SQL). In our testing, the numbers in the answers stayed the same between runs, but the wording of the answers still varied.

---

## SQL Toolkit and Tools <a name="agent-toolkit"></a>

We then create a toolkit that gives the agent SQL-specific abilities:

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name)
    print(" ", tool.description)
    print()
```

```
sql_db_query
  Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.

sql_db_schema
  Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3

sql_db_list_tables
  Input is an empty string, output is a comma-separated list of tables in the database.

sql_db_query_checker
  Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!
```
<br>
These tools allow the agent to; inspect which tables exist, look at table info and sample rows, check a query before running it, and execute it.

The LLM never sees the code behind a tool, only its name and description, so the descriptions act as instructions. They tell the agent to list tables before reading a schema, to check a query before running it, and to rewrite a query if it returns an error. `sql_db_query_checker` is the only tool that uses the LLM itself, which is why the toolkit needs `llm=llm`.

Rather than guessing SQL from scratch, the agent can actively *work with* the database.

---

## System Prompt <a name="agent-prompt"></a>

The system prompt is critical. It defines the agent’s role, scope, and guardrails.

We read it from a separate text file:

```python
with open(config.PROMPT_FILE, "r", encoding="utf-8") as f:
    system_text = f.read()
```
<br>
For reference, the content of *sql-agent-system-prompt.txt* is shown below:

```text
ROLE:

You are an expert SQL data analyst, and helpful assistant. Convert the user’s question into an accurate, efficient PostgreSQL 16 SELECT query and a concise natural-language answer.


SCOPE:

- Use read-only SELECT only (never INSERT/UPDATE/DELETE/DDL/admin).

- All tables live in schema grocery_db. Always schema-qualify (e.g grocery_db.customer_details)


TABLE INFORMATION:

1. grocery_db.customer_details

grocery_db.customer_details holds data at the level of one row per unique customer_id

Columns:

- customer_id INT (the unique identifier for the customer)
- distance_from_store NUMERIC(10,2) (the distance the customer lives from the store, in miles)
- gender VARCHAR(2) (values include 'M', 'F', and NULL)
- credit_score NUMERIC(3,2) (a decimal value between 0.00 and 1.00 pertaining to the customer's credit score)

2. grocery_db.transactions

grocery_db.transactions holds data at the level of one row per customer_id, transaction_id, product_area_id

Columns:

- customer_id INT (the unique identifier for the customer)
- transaction_date DATE (the date of the transaction, for example 2020-04-10)
- transaction_id INT (a unique id for each individual transaction)
- product_area_id INT (a number from 1 to 5 that represents the product area that was shopped in. There can be multiple product areas within a transaction)
- num_items INT (the number items within that product area, for that transaction)
- sales_cost NUMERIC(10,2) (the monetary value for the items purchased within that product area, for that transaction)


TABLE JOIN RELATIONSHIPS:

grocery_db.customer_details can be joined to grocery_db.transactions using the shared customer_id column


DATA WINDOW:

Transactions data is available between 2020-04-01 to 2020-09-30. If the user asks about "this period", assume that range. If they ask about a different period, filter explicitly. If they ask for data from outside this window, reply to them with the information pertaining to the period that is available.


QUERY DESIGN RULES:

SELECT-only. If returning raw rows, add LIMIT 100.

Aggregation discipline.

"Number of customers" = COUNT(DISTINCT customer_id) at the appropriate filter.

"Number of transactions/visits" = COUNT(DISTINCT transaction_id).

Revenue and Items = SUM(sales_cost) and SUM(num_items) at the right grouping.

Avoid double-counting: the transactions table is at the level of multiple product_area_id's per transaction_id.

Categoricals: Use actual domain values shown in samples (for example, gender IN ('M','F')). If uncertain, first check SELECT DISTINCT ... LIMIT 10.

Performance & clarity. Select only needed columns; use CTEs for readability; alias columns clearly; round monetary outputs for readability (for example, ROUND(SUM(sales_cost), 2)).

Ambiguity. If a question is ambiguous (for example, "top products" but no timeframe), ask a brief clarifying question before querying.



EXAMPLE QUERIES & RESPONSE FOR GUIDANCE:

Question A: 

"How many customers are male?"

SQL Query A:

select
  count(*) as num_custs
  
from 
  grocery_db.customer_details
  
where
  gender = 'M';

Response A:

"There are 380 male customers"


Question B: 

"What is the average credit score, by gender?"

SQL Query B:

select
  gender,
  avg(credit_score) as average_credit_score
  
from
  grocery_db.customer_details
  
group by
  gender;

Response B:

"The average credit score by gender is as follows:

- Female (F): 0.601
- Male (M): 0.593
- Unspecified: 0.563"


Question C: 

"Which customer had the highest average transaction value in July 2020, and what was that value?"

SQL Query B:

with transaction_values as (
select
  customer_id,
  transaction_id,
  sum(sales_cost) as transaction_value
  
from
  grocery_db.transactions
  
where
  transaction_date >= '2020-07-01'
  and transaction_date <=  '2020-07-31'
  
group by
  customer_id,
  transaction_id

),

avg_per_customer as (
select
  customer_id,
  avg(transaction_value) as avg_transaction_value
  
from
  transaction_values
  
group by
  customer_id
  
)

select
  customer_id,
  round(avg_transaction_value, 2) as avg_transaction_value
  
from
  avg_per_customer
  
order by
  avg_transaction_value desc,
  customer_id
  
limit 1;

Response C:

"Customer 514 had the highest average transaction value in July 2020, at $1027.77"
```
<br>
This prompt gives the agent:

* A clear role
* Strict scope and safety rules
* Table descriptions and relationships
* Timing constraints
* Query design rules
* A small number of worked examples

All of which increase the chance of correct queries.

Note that the "SELECT only" rule lives in the prompt: the `sql_db_query` tool will run whatever SQL it's given. In production, connecting with a read-only database user would make sure the agent couldn't change data even if it ignored the prompt.

---

## Creating the Agent <a name="agent-create"></a>

Finally, we create the agent itself:

```python
from langchain.agents import create_agent

agent = create_agent(model=llm,
                     tools = tools,
                     system_prompt=system_text)
```

The agent now has:

* The LLM (*llm*)
* The SQL tools (*tools*)
* The system prompt (*system_text*)

Given a user question, it can plan, call tools, and reason step by step toward a final answer.

___

# 05. Application & Examples <a name="agent-application"></a>

To send a query to the agent, we use LangChain’s *HumanMessage* format:

```python
from langchain_core.messages import HumanMessage

user_query = "On average, which gender lives furthest from store?"

result = agent.invoke({"messages": [HumanMessage(content=user_query)]})
print(result["messages"][-1].content)
```

`result["messages"]` holds every step the agent took: the question, each tool call and its result, and the final answer. We printed all of them for each question, and then checked the answer with our own SQL.

Two example questions are seen below:

1. **Question:** *On average, which gender lives furthest from the store?*

```
On average, female customers (gender 'F') live furthest from the store, with an average distance of 3.98 miles.
```

The agent:

* Recognised this is a question about customer-level data
* Used the *customer_details* table
* Computed average distance by gender, including customers with no gender recorded
* Went straight to `sql_db_query`, skipping the other tools, because the system prompt already describes the table

The query it wrote:

```sql
select
  gender,
  round(avg(distance_from_store), 2) as avg_distance_from_store
from
  grocery_db.customer_details
group by
  gender
order by
  avg_distance_from_store desc;
```

We verified this by running our own query on the database:

```python
check_query = """
select
  gender,
  round(avg(distance_from_store), 2) as avg_distance,
  count(*) as num_customers
from
  grocery_db.customer_details
group by
  gender
order by
  avg_distance desc;
"""

print(db.run(check_query))
```

```
[('F', Decimal('3.98'), 485), (None, Decimal('3.36'), 5), ('M', Decimal('0.87'), 380)]
```

Females average 3.98 miles (485 customers), customers with no gender recorded 3.36 (5 customers), and males 0.87 (380 customers), so the agent's answer is correct.

2. **Question:** *What is the average transaction value in September 2020 for male customers who have a credit score above 0.5?*

```
The average transaction value in September 2020 for male customers with a credit score above 0.5 was $88.49.
```

This required both tables and more careful logic. Here, the agent:

* Used `sql_db_query_checker` before running the query this time (the checker returned it unchanged)
* Filtered customers by gender and credit score
* Joined to the *transactions* table, filtered to September 2020
* Correctly aggregated sales at the transaction level (summing by *transaction_id* first)
* Then averaged these transaction values

The query it wrote:

```sql
with filtered_customers as (
  select customer_id
  from grocery_db.customer_details
  where gender = 'M' and credit_score > 0.5
),
transaction_values as (
  select
    t.transaction_id,
    sum(t.sales_cost) as transaction_value
  from
    grocery_db.transactions t
    join filtered_customers fc on t.customer_id = fc.customer_id
  where
    t.transaction_date >= '2020-09-01'
    and t.transaction_date <= '2020-09-30'
  group by
    t.transaction_id
)
select
  round(avg(transaction_value), 2) as avg_transaction_value
from
  transaction_values;
```

"Average transaction value" can be read in two ways, so we checked both:

* **A:** the average over all transactions
* **B:** each customer's average transaction value first, then the average of those

```python
print("A (all transactions):     ", db.run(check_a))
print("B (average per customer): ", db.run(check_b))
```

```
A (all transactions):      [(Decimal('88.49'), 1046)]
B (average per customer):  [(Decimal('101.32'), 277)]
```

The agent's answer matches reading A exactly ($88.49 over 1,046 transactions), so its SQL was correct.

However, the question was ambiguous, and the agent chose reading A without saying so. Reading B is 14% higher, because customers who shop more often tend to spend less per trip, and reading A gives them more weight. The system prompt asks the agent to ask a clarifying question when a question is ambiguous, but it didn't treat this one as ambiguous.

These examples demonstrate that the agent is not just writing plausible SQL, but is **reasoning correctly about grain, joins, and aggregations**, guided by the system prompt and tools. They also show that a correct query can still answer a different question from the one the user meant, which is why checking the answers matters.

___

# 06. Growth & Next Steps <a name="growth-next-steps"></a>

Potential future enhancements include:

* Exposing additional tables and relationships as the data model grows
* Adding a light web UI so business users can ask questions without touching SQL
* Logging queries and answers for audit, training, and documentation
* Adding automated evaluations to catch incorrect queries or edge cases
* Allowing the agent to ask clarification questions in more complex scenarios

This project provides a foundation for an AI-powered, self-serve analytics layer on top of ABC Grocery’s SQL data.

The full code is available on [GitHub](https://github.com/lemuelhmgs12/sql_agent_bot).

___