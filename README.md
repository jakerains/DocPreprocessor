# Docling Document Processor

A Streamlit-based web application for processing documents using Docling, converting them into LLM-ready formats.

## Features

- Support for multiple document formats (PDF, Word, PowerPoint, Excel, HTML, Markdown)
- Export to JSON or Markdown formats
- Multiple parsing modes (Default, Semantic, Structure)
- Clean and intuitive user interface
- Download processed documents

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jakerains/docling.git
cd docling
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install streamlit docling
```

## Usage

1. Start the Streamlit application:
```bash
cd src
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload a document and select your processing options

4. View and download the processed results

## Project Structure

```
docling/
├── docs/
│   ├── CHANGELOG.md
│   └── requirements.md
├── src/
│   ├── app.py
│   ├── static/
│   └── templates/
└── venv/
```

## Documentation

- [Requirements](docs/requirements.md)
- [Changelog](docs/CHANGELOG.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Jake Rains (@jakerains) 