# KnowledgeVault

EDGAR-first, local-first RAG stack:
1) Fetch SEC filings → 2) Convert to .txt → 3) Embed + index → 4) Ask (local LLM) → 5) Extract metrics.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SEC_USER_AGENT="KnowledgeVault/0.1 (your.email@example.com)"

# Fetch a few filings
python scripts/kv_fetch_edgar.py --tickers MQ --forms 10-K 10-Q --limit 3

# Build index
python plugins/knowledgevault/scripts/kv_ingest.py data/edgar/txt data/context --index-dir plugins/knowledgevault/data/index

# Ask
ollama pull llama3.2:3b-instruct-q4_0
python plugins/knowledgevault/scripts/kv_ask.py "Summarize revenue and profitability for the latest fiscal year." --index-dir plugins/knowledgevault/data/index

# Extract metrics
python plugins/knowledgevault/scripts/kv_ask_metrics.py --company "Marqeta MQ" --index-dir plugins/knowledgevault/data/index
```
