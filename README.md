# Data Insights Agent

A comprehensive data analysis and insights generation tool powered by LangChain and vector embeddings.

## Project Structure

```
data-insights-agent/
├── data/                    # Data storage
│   ├── raw/                # Raw datasets
│   └── processed/          # Processed datasets
├── src/                    # Source code
│   ├── data_loader.py      # Load and preprocess datasets
│   ├── analytics.py        # Basic analyses, summaries, charts
│   ├── embeddings.py       # Generate embeddings for text units
│   ├── vector_store.py     # FAISS integration (index, search, save/load)
│   ├── agent.py            # LangChain-based retrieval-augmented agent
│   ├── prompts.py          # System and user prompts for the agent
│   ├── utils.py            # Helpers (logging, config)
│   ├── report.py           # Markdown/HTML report generation
│   └── ui.py               # Streamlit/CLI interface
├── app/                    # Application entry points
│   └── run.py              # Entry point for CLI or UI
├── tests/                  # Test files
│   ├── test_data_functions.py
│   └── test_vector_store.py
├── notebooks/              # Jupyter notebooks
│   └── 01_exploration.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Features

- **Data Loading & Preprocessing**: Flexible data loading with preprocessing capabilities
- **Analytics Engine**: Basic statistical analyses, summaries, and chart generation
- **Vector Embeddings**: Generate embeddings for text data using various models
- **Vector Store**: FAISS-based vector storage and similarity search
- **AI Agent**: LangChain-powered retrieval-augmented generation for insights
- **Report Generation**: Automated Markdown/HTML report creation
- **User Interface**: Both Streamlit web interface and CLI support

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app/run.py`

## Usage

### CLI Mode

```bash
python app/run.py --mode cli --data-path data/raw/your_dataset.csv
```

### Web Interface

```bash
python app/run.py --mode web
```

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Features

1. Create feature branch
2. Implement functionality in appropriate `src/` module
3. Add tests in `tests/`
4. Update documentation
5. Submit pull request

## License

MIT License
