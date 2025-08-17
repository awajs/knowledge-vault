# Knowledge Vault

**Knowledge Vault** is an experimental pipeline for extracting
structured financial metrics from SEC EDGAR filings (10-K / 10-Q) using
a mix of deterministic parsing, retrieval, and local LLM inference (via
[Ollama](https://ollama.ai/)).

------------------------------------------------------------------------

## Features

-   **EDGAR fetcher**: Download 10-K and 10-Q filings by ticker symbol
    into a local folder structure.
-   **Indexing**: Store filings in `data/edgar/txt/<TICKER>/...` for
    later querying.
-   **RAG-style querying** (`kv_ask.py`):
    -   Retrieves relevant filing text.\
    -   Asks free-form questions (e.g., *"Summarize revenue and
        profitability for the latest fiscal year"*).\
    -   Returns structured JSON with citations.\
-   **Metric extraction** (`kv_ask_metrics.py`):
    -   Extracts **Revenue, Cost of revenue, Operating income, Net
        income, EPS (basic/diluted)** from the latest annual
        consolidated income statement.\
    -   Uses a **hybrid approach**: deterministic table parsing first,
        then LLM fallback if needed.\
    -   Guardrails: filters out percent-of-revenue tables, stock-based
        comp notes, and junk footnotes.\
    -   Scales numbers properly based on "in millions / in thousands"
        headers.\
-   **Ollama integration**: Run small/medium models locally (e.g.,
    `mistral:7b-instruct`, `llama3.1:8b-instruct-q4_0`).

------------------------------------------------------------------------

## Requirements

-   Python 3.10+ (tested in venv)
-   Dependencies: `pip install -r requirements.txt`\
    (e.g., `requests`, `numpy`, `pandas`, etc. --- add as needed)
-   [Ollama](https://ollama.ai/) installed and running
-   NVIDIA GPU recommended (8 GB VRAM fits 7B models fine; CPU also
    works slower)

------------------------------------------------------------------------

## Setup

1.  **Clone repository**:

    ``` bash
    git clone https://github.com/<your-org>/knowledge-vault.git
    cd knowledge-vault
    ```

2.  **Create virtual environment**:

    ``` bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Start Ollama** in background:

    ``` bash
    ollama serve >/dev/null 2>&1 &
    ```

4.  **Pull models** (choose one that fits your GPU/CPU):

    ``` bash
    ollama pull mistral:7b-instruct
    ollama pull llama3.1:8b-instruct-q4_0
    ```

------------------------------------------------------------------------

## Usage

### 1. Fetch filings

Fetch latest 10-K / 10-Q for one or more tickers:

``` bash
python scripts/kv_fetch_edgar.py --tickers NVDA AAPL MQ --forms 10-K 10-Q --limit 2
```

Files are saved under `data/edgar/txt/<TICKER>/`.

------------------------------------------------------------------------

### 2. Ask free-form question

``` bash
python plugins/knowledgevault/scripts/kv_ask.py   "Summarize revenue and profitability for the latest fiscal year."   --index-dir data/edgar/txt --k 12 --model mistral:7b-instruct
```

------------------------------------------------------------------------

### 3. Extract structured metrics

``` bash
python plugins/knowledgevault/scripts/kv_ask_metrics.py   --company NVDA   --index-dir data/edgar/txt   --model mistral:7b-instruct   --debug
```

Output:

``` json
{
  "answer": "OK",
  "metrics": [
    {"name": "netRevenue", "value": 130497000000.0, "unit": "USD"},
    {"name": "costOfRevenue", "value": 32639000000.0, "unit": "USD"},
    {"name": "operatingIncome", "value": 81453000000.0, "unit": "USD"},
    {"name": "netIncome", "value": 72880000000.0, "unit": "USD"},
    {"name": "basicEPS", "value": 2.97, "unit": "USD/share"},
    {"name": "dilutedEPS", "value": 2.94, "unit": "USD/share"}
  ],
  "period": {"startDate": "Jan 26, 2025", "endDate": "Jan 28, 2024"},
  "citations": [...]
}
```

------------------------------------------------------------------------

## Current Challenges

-   OCR-like tables: SEC text files are messy (pipes `|`, multi-year
    columns, percentage tables).\
-   Small models sometimes misinterpret table context. Guardrails help,
    but scaling up to 7B--13B models improves robustness.\
-   Next steps include:
    -   Better date-column alignment (ensuring the latest FY column is
        always selected).\
    -   Optional training/reranking models for more robust table
        selection.\
    -   Extending to balance sheets and cash flow statements.

------------------------------------------------------------------------

## Roadmap

-   ✅ Income statement extraction\
-   ⬜ Balance sheet / cash flow metrics\
-   ⬜ Multi-company batch comparisons\
-   ⬜ Lightweight LoRA training for specialized table parsing\
-   ⬜ Streamlit dashboard for visualization
