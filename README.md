# OLDP Toolkit

A set of command-line tools and scripts for processing data from the Open Legal Data project.

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup environment

```bash
# Create new virtual environment for Python 3.12
uv venv --python=3.12

# Activate environment
source .venv/bin/activate
```

### Install toolkit

#### Pip

```bash
pip install git+https://github.com/openlegaldata/oldp-toolkit.git#egg=oldp_toolkit
```

#### Local

1. Clone the repository:
```bash
git clone https://github.com/openlegaldata/oldp-toolkit.git
cd oldp-toolkit
```

2. Install dependencies and the package:
```bash
make install
```

Alternatively, you can install manually:
```bash
uv sync --dev
uv pip install -e .
```

## Usage

The toolkit provides the `oldpt` command with various subcommands:

```bash
# Show help
oldpt --help

# Enable debug logging
oldpt --debug <command>

# Examples: Convert JSONL dump to different formats

# Upload to HuggingFace Hub
oldpt convert_dump_to_hf input.jsonl username/dataset-name --format hf_hub --config-name default --limit 1000 --private

# Save as JSONL file, skipping first 100 entries
oldpt convert_dump_to_hf input.jsonl output.jsonl --format jsonl --skip 100

# Save as Parquet file with skip and limit
oldpt convert_dump_to_hf input.jsonl output.parquet --format parquet --skip 500 --limit 1000

# Save to disk in HF format with custom split
oldpt convert_dump_to_hf input.jsonl ./dataset --format hf_disk --config-name legal_cases --split validation
```

### Available Commands

- `convert_dump_to_hf`: Convert JSONL dumps to various formats (HF Hub, JSONL, Parquet, HF disk format)

#### Convert Dump to HF Command Details

The `convert_dump_to_hf` command supports multiple output formats:

1. **HuggingFace Hub** (`--format hf_hub`): Upload dataset directly to HuggingFace Hub
   - Requires HF authentication (`huggingface-cli login`)
   - Supports private repositories with `--private` flag
   - **Requires** `--config-name` argument
   - Uses `--split` argument (default: "train")

2. **JSONL** (`--format jsonl`): Save as JSONL file
   - Supports both local and remote filesystems (e.g., S3, GCS)
   - Uses HuggingFace datasets framework for remote support

3. **Parquet** (`--format parquet`): Save as Parquet file
   - Efficient columnar format for large datasets
   - Supports both local and remote filesystems

4. **HF Disk** (`--format hf_disk`): Save in HuggingFace dataset format to disk
   - Preserves dataset structure and metadata
   - Can be loaded later with `datasets.load_from_disk()`
   - **Requires** `--config-name` argument
   - Uses `--split` argument (default: "train")
   - Creates structured directory: `output_dir/config_name/split/`

**Additional Features:**
- HTML to Markdown conversion (use `--no-process` to skip)
- Legal reference extraction using RefExtractor
- Support for gzipped input files
- Configurable batch processing
- Skip entries with `--skip N` (skip first N entries)
- Limit dataset size with `--limit N` (applied after skip)

## Development

### Running Tests

```bash
make test
```

### Linting and Formatting

```bash
# Check linting
make lint-check

# Format code with ruff
make lint
```

### Pre-commit Hooks

The project includes pre-commit hooks for automatic code formatting and linting:

```bash
# Install pre-commit hooks
make pre-commit-install
```

The hooks will automatically run ruff formatting and linting checks before each commit.

### Project Structure

```
oldp-toolkit/
├── src/
│   └── oldp_toolkit/
│       ├── __init__.py
│       ├── cli.py              # Main CLI entrypoint
│       └── commands/
│           ├── __init__.py
│           ├── base.py         # BaseCommand class with colored logging
│           └── convert_dump_to_hf.py  # Convert JSONL to HF formats
├── tests/
│   ├── test_cli.py
│   ├── test_base_command.py
│   └── test_convert_dump_to_hf.py
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── pyproject.toml              # Project configuration
├── Makefile                    # Build automation
└── README.md
```

## Adding New Commands

1. Create a new command file in `src/oldp_toolkit/commands/`
2. Inherit from `BaseCommand` and implement required methods
3. Register the command in `cli.py`
4. Add tests in `tests/`

Example:
```python
from oldp_toolkit.commands.base import BaseCommand

class MyCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("--my-arg", help="My argument")
    
    def handle(self, args):
        logger = logging.getLogger(__name__)
        logger.info("Executing my command")
        # Implementation here
```

## License

MIT