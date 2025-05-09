import streamlit as st
import pandas as pd
import sqlite3
import openai
import json
import xmltodict
import pdfplumber
import requests
from io import StringIO, BytesIO
import re

# ---- Page Configuration ----
st.set_page_config(page_title="📊 AI Data Chatbot", layout="wide")

# ---- Sidebar for API Keys ----
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
    groq_key = st.text_input("Groq API Key", type="password", key="groq_key")

    if api_key:
        openai.api_key = api_key
        st.success("✅ OpenAI API Key set!")

    if groq_key:
        st.success("✅ Groq API Key set!")

    st.markdown("---")
    st.markdown("### About")
    st.info("Upload your data and chat with it using AI or SQL!")

# ---- Main UI ----
st.title("📊 AI Data Chatbot")
st.markdown("Upload your data file and start asking questions in natural language or SQL!")

# ---- DB Connection ----
def connect_to_db():
    return sqlite3.connect('data.db')

# ---- Clean list fields ----
def clean_lists_in_dataframe(df):
    def stringify(x):
        if isinstance(x, list):
            return ', '.join([str(item) if not isinstance(item, dict) else json.dumps(item) for item in x])
        elif isinstance(x, dict):
            return json.dumps(x)
        else:
            return x

    for col in df.columns:
        df[col] = df[col].apply(stringify)
    return df

# ---- Flatten deeply nested XML ----
def flatten_xml(data_dict):
    def recurse(data):
        if isinstance(data, list):
            return [recurse(item) for item in data]
        elif isinstance(data, dict):
            return {k: recurse(v) for k, v in data.items()}
        else:
            return data
    return recurse(data_dict)

# ---- Get sample values safely ----
def get_safe_sample(series, n=3):
    """Get sample values from a series, handling empty series safely"""
    # Drop NA values
    non_null = series.dropna()
    
    # If there are no non-null values, return empty list
    if len(non_null) == 0:
        return ["No non-null values"]
    
    # If there are fewer values than requested, return all
    if len(non_null) <= n:
        return non_null.tolist()
    
    # Otherwise sample n values
    return non_null.sample(n).tolist()

# ---- PDF Processing ----
def process_pdf(file):
    # Extract text from PDF
    full_text = ""
    sections = []
    current_section = {"title": "Introduction", "content": ""}

    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n"

                # Try to identify section headers
                lines = text.split('\n')
                for line in lines:
                    # Check if line is likely a section header (short, capitalized)
                    stripped_line = line.strip()
                    if len(stripped_line) < 50 and stripped_line and not stripped_line[0].isdigit():
                        # If it looks like a header, start a new section
                        if current_section["content"]:
                            sections.append(current_section)
                        current_section = {"title": stripped_line, "content": ""}
                    else:
                        current_section["content"] += line + "\n"

    # Add the last section
    if current_section["content"]:
        sections.append(current_section)

    # Create a DataFrame from the sections
    df = pd.DataFrame(sections)

    # Also create a DataFrame with page-by-page content
    pages_df = pd.DataFrame({
        "page_num": range(1, len(pdf.pages) + 1),
        "page_content": [page.extract_text() or "" for page in pdf.pages]
    })

    # Store the full text in session state for semantic search
    st.session_state.pdf_full_text = full_text

    return df, pages_df

# ---- File Parsing ----
def parse_file(file):
    file_type = file.name.split('.')[-1].lower()

    if file_type == 'csv':
        return pd.read_csv(file)

    elif file_type == 'xlsx' or file_type == 'xls':
        try:
            # Force pandas to use openpyxl and never fall back to xlrd
            import pandas
            # Temporarily disable the xlrd import
            original_import = pandas.io.excel._base.import_optional_dependency

            def mock_import(name, **kwargs):
                if name == "xlrd":
                    return None
                return original_import(name, **kwargs)

            # Replace the import function
            pandas.io.excel._base.import_optional_dependency = mock_import

            # Now read the Excel file
            result = pd.read_excel(file, engine='openpyxl')

            # Restore the original import function
            pandas.io.excel._base.import_optional_dependency = original_import

            return result
        except Exception as e:
            st.error(f"Excel reading error: {str(e)}")
            return None

    elif file_type == 'json':
        data = json.load(file)
        return pd.json_normalize(data)

    elif file_type == 'xml':
        xml_content = file.read()
        try:
            data_dict = xmltodict.parse(xml_content)
            flattened = flatten_xml(data_dict)
            return pd.json_normalize(flattened)
        except Exception as e:
            st.error(f"❌ XML parsing failed: {e}")
            return None

    elif file_type == 'pdf':
        # Special handling for PDFs
        st.session_state.is_pdf = True
        sections_df, pages_df = process_pdf(file)

        # Store both dataframes in session state
        st.session_state.pdf_sections = sections_df
        st.session_state.pdf_pages = pages_df

        # Return the sections dataframe as the main one
        return sections_df

    else:
        st.warning("Unsupported file format.")
        return None

# ---- Insert into DB ----
def insert_data_to_db(df):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS data_table")
    conn.commit()

    df = clean_lists_in_dataframe(df)
    df.to_sql('data_table', conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()

# ---- Query DB ----
def query_db(query):
    conn = connect_to_db()
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result

# ---- Natural Language to SQL ----
def get_sql_from_prompt(user_input, columns):
    quoted_columns = [f'"{col}"' for col in columns]

    prompt = f"""
You are a data analyst. The user uploaded a dataset with these columns:
{quoted_columns}.

Convert the user's natural language question into a valid SQLite-compatible SQL query.
Only return the SQL query string, no explanation.

Question: {user_input}
"""

    if api_key:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response['choices'][0]['message']['content'].strip().strip("```sql").strip("```")
        except Exception as e:
            return f"❌ OpenAI Error: {str(e)}"

    elif groq_key:
        try:
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            data = {
                "messages": [
                    {"role": "system", "content": "You are an AI assistant that converts questions into SQL queries for a SQLite table named 'data_table'."},
                    {"role": "user", "content": prompt}
                ],
                "model": "llama3-70b-8192"
            }
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

            if res.status_code != 200:
                return f"❌ Groq API Error {res.status_code}: {res.text}"

            response_json = res.json()
            if "choices" in response_json:
                return response_json['choices'][0]['message']['content'].strip().strip("```sql").strip("```")
            else:
                return f"❌ Groq Error: Unexpected response format: {response_json}"

        except Exception as e:
            return f"❌ Groq Exception: {str(e)}"

    else:
        return "API_KEY_REQUIRED"

# ---- Semantic Search for PDFs ----
def semantic_search_pdf(query, full_text):
    if not api_key and not groq_key:
        return "⚠️ Please enter an OpenAI or Groq API key in the sidebar to use semantic search."

    prompt = f"""
You are an AI assistant helping with document search. The user has uploaded a PDF document and is asking a question about it.
Based on the document content below, provide a concise and accurate answer to the user's question.
If the answer is not in the document, say "I don't see information about that in the document."

Document content:
{full_text[:10000]}  # Limit to first 10000 chars to avoid token limits

User question: {query}

Your answer should be:
1. Directly relevant to the question
2. Based only on information in the document
3. Concise (1-3 paragraphs)
4. Include specific details from the document when available
"""

    if api_key:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ OpenAI Error: {str(e)}"

    elif groq_key:
        try:
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            data = {
                "messages": [
                    {"role": "system", "content": "You are an AI assistant that helps answer questions about documents."},
                    {"role": "user", "content": prompt}
                ],
                "model": "llama3-70b-8192"
            }
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

            if res.status_code != 200:
                return f"❌ Groq API Error {res.status_code}: {res.text}"

            response_json = res.json()
            if "choices" in response_json:
                return response_json['choices'][0]['message']['content']
            else:
                return f"❌ Groq Error: Unexpected response format: {response_json}"

        except Exception as e:
            return f"❌ Groq Exception: {str(e)}"

# ---- Detect if input is SQL ----
def is_sql_query(text):
    # Simple heuristic to detect if text is likely SQL
    sql_keywords = ['select', 'from', 'where', 'group by', 'order by', 'having', 'join', 'inner join', 'left join']
    text_lower = text.lower()

    # Check if it starts with SELECT
    if text_lower.strip().startswith('select'):
        return True

    # Check if it contains multiple SQL keywords
    keyword_count = sum(1 for keyword in sql_keywords if keyword in text_lower)
    if keyword_count >= 2:
        return True

    return False

# ---- Chat History ----
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Flag for PDF files
if 'is_pdf' not in st.session_state:
    st.session_state.is_pdf = False

# Flag for input clearing
if 'clear_input' not in st.session_state:
    st.session_state.clear_input = False

# ---- File Upload ----
# --- Safe reset for file uploader state on Streamlit Cloud ---
if 'file_uploader' in st.session_state:
    del st.session_state['file_uploader']

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "json", "xml", "pdf"], key="file_uploader")

# ---- Main Logic ----
if uploaded_file:
    # Process the file
    if 'df' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
        with st.spinner("Processing your file..."):
            df = parse_file(uploaded_file)

            if df is not None:
                st.session_state.df = df
                st.session_state.last_file = uploaded_file.name

                # Only insert into DB if not a PDF
                if not st.session_state.get('is_pdf', False):
                    insert_data_to_db(df)

                st.success("✅ File processed successfully!")

                # Display a sample of the data
                st.subheader("Data Preview")
                st.dataframe(df.head())

                # Show column information - Fix for PyArrow error
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': [str(dtype) for dtype in df.dtypes],  # Convert dtype objects to strings
                    'Non-Null Count': df.count().tolist(),
                    'Sample Values': [str(get_safe_sample(df[col])) for col in df.columns]  # Use safe sampling
                })
                st.dataframe(col_info)

                # If it's a PDF, show additional information
                if st.session_state.get('is_pdf', False):
                    st.info("📄 PDF detected! You can ask questions about the content of the document.")

                    # Show the sections found
                    with st.expander("Document Sections"):
                        for i, row in df.iterrows():
                            st.markdown(f"**{row.get('title', 'No Title')}**")

                            st.markdown(f"**{row['title']}**")
                            st.text(row['content'][:200] + "..." if len(row['content']) > 200 else row['content'])
            else:
                st.error("❌ Failed to parse file.")
    else:
        df = st.session_state.df
        st.success("✅ Using previously loaded data")

        # Display a sample of the data
        with st.expander("Data Preview", expanded=False):
            st.dataframe(df.head())

    # Chat interface
    st.markdown("---")
    st.subheader("💬 Chat with your data")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")
                if 'data' in message:
                    st.dataframe(message['data'])

    # Create a unique key for each session to force input field recreation
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

    # Input for new message with dynamic key
    user_input = st.text_input("Ask a question about your data", key=f"user_input_{st.session_state.input_key}")

    # Process button
    if st.button("Send") and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})

        # Process the query
        with st.spinner("Processing your question..."):
            # Check if we're dealing with a PDF
            if st.session_state.get('is_pdf', False):
                # Use semantic search for PDFs
                if not api_key and not groq_key:
                    response = "⚠️ Please enter an OpenAI or Groq API key in the sidebar to ask questions about PDF documents."
                else:
                    response = semantic_search_pdf(user_input, st.session_state.pdf_full_text)

                # Add response to chat history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })

            # For non-PDF files, use the existing SQL approach
            else:
                # Check if it's a SQL query
                if is_sql_query(user_input):
                    try:
                        # Execute SQL directly
                        result = query_db(user_input)

                        # Add response to chat history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': f"Here's the result of your SQL query:",
                            'data': result
                        })
                    except Exception as e:
                        # Add error to chat history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': f"❌ Error executing SQL: {str(e)}"
                        })
                else:
                    # Natural language query
                    sql_query = get_sql_from_prompt(user_input, list(df.columns))

                    if sql_query == "API_KEY_REQUIRED":
                        # Add error to chat history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': "⚠️ Please enter an OpenAI or Groq API key in the sidebar to use natural language queries."
                        })
                    elif "❌" in sql_query:
                        # Add error to chat history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': sql_query
                        })
                    else:
                        try:
                            # Execute the generated SQL
                            result = query_db(sql_query)

                            # Add response to chat history
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': f"I converted your question to SQL: \n```sql\n{sql_query}\n```\n\nHere's the result:",
                                'data': result
                            })
                        except Exception as e:
                            # Add error to chat history
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': f"I generated this SQL: \n```sql\n{sql_query}\n```\n\n❌ But there was an error: {str(e)}"
                            })

        # Increment the input key to force recreation of the input field (clearing it)
        st.session_state.input_key += 1

        # Use st.rerun() to update the UI
        st.rerun()

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.info("👆 Please upload a file to get started!")

    # Show sample queries
    st.markdown("### Sample queries you can try after uploading data:")
    st.markdown("""
    - "What is the total revenue by segment?"
    - "Show me the top 5 customers by order value"
    - "What's the average price per product category?"
    - "SELECT * FROM data_table LIMIT 10"

    For PDF documents:
    - "What is this document about?"
    - "How long does it take to learn Python according to the roadmap?"
    - "What are the essential concepts for machine learning?"
    """)
