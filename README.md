# AI Data Chatbot

This project is a Streamlit-based web application that allows users to upload various data files and interact with their data using natural language queries or SQL queries. The app supports multiple file formats including CSV, Excel, JSON, XML, and PDF.

## Features

- Upload data files in CSV, XLSX, JSON, XML, or PDF formats.
- Parse and display uploaded data.
- Store data in a local SQLite database.
- Ask natural language questions about the data, which are converted to SQL queries using OpenAI or Groq APIs.
- Run dynamic predefined queries for common questions.
- Execute custom SQL queries using DuckDB syntax.
- Display query results in a user-friendly table.

## Installation

1. Clone the repository or download the source code.

2. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open the app in your browser (usually at `http://localhost:8501`).

3. Enter your OpenAI or Groq API key (optional) to enable natural language to SQL conversion.

4. Upload your data file.

5. Ask questions about your data or write your own SQL queries.

## File Formats Supported

- CSV
- Excel (XLSX)
- JSON
- XML
- PDF (text extraction)

## Notes

- The app uses SQLite as the local database to store uploaded data.
- Natural language queries are converted to SQL using OpenAI GPT-4o or Groq API.
- If no API key is provided, only predefined dynamic queries and manual SQL queries are supported.

## Dependencies

See `recuirment.txt` for the list of required Python packages.

## License

This project is provided as-is without any warranty.
