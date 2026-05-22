"""Convert dump to HuggingFace format command."""

import gzip
import json
import logging
from pathlib import Path

from datasets import Dataset, Features, Value
from markdownify import markdownify as md
from refex.extractor import RefExtractor, RefMarker

from oldp_toolkit.commands.base import BaseCommand

logger = logging.getLogger(__name__)


SUPPORTED_TYPES = ("cases", "laws", "references")


# Explicit HF features schema for the *raw* cases JSONL (before
# process_case adds ``markdown_content`` and ``reference_markers``).
#
# Required for ``Dataset.from_generator``: PyArrow infers each column's
# type from the first batch only. For optional nested fields in the
# ``court`` struct (``city``, ``jurisdiction``, ``level_of_appeal``)
# the first batch's null values get typed as Arrow ``null``, and a
# later non-null value (e.g. ``city=4``) fails to cast — manifesting
# as ``DatasetGenerationError("An error occurred while generating the
# dataset")`` mid-stream. Pre-declaring the schema makes the type a
# nullable int / string from the start.
CASES_FEATURES = Features({
    "id":           Value("int64"),
    "slug":         Value("string"),
    "court": {
        "id":              Value("int64"),
        "name":            Value("string"),
        "slug":            Value("string"),
        "city":            Value("int64"),
        "state":           Value("int64"),
        "jurisdiction":    Value("string"),
        "level_of_appeal": Value("string"),
    },
    "file_number":  Value("string"),
    "date":         Value("string"),
    "created_date": Value("string"),
    "updated_date": Value("string"),
    "type":         Value("string"),
    "ecli":         Value("string"),
    "content":      Value("string"),
})


# Explicit HF features schema for the *raw* laws JSONL (before
# process_law adds ``markdown_content``). Same rationale as
# ``CASES_FEATURES``: nullable string columns (``amtabk``, ``kurzue``,
# ``doknr``) first appear as ``None`` and would be typed as Arrow
# ``null`` by inference, then later non-null values would fail to cast.
LAWS_FEATURES = Features({
    "id":           Value("int64"),
    "book":         Value("int64"),
    "book_code":    Value("string"),
    "book_slug":    Value("string"),
    "title":        Value("string"),
    "content":      Value("string"),
    "slug":         Value("string"),
    "created_date": Value("string"),
    "updated_date": Value("string"),
    "section":      Value("string"),
    "amtabk":       Value("string"),
    "kurzue":       Value("string"),
    "doknr":        Value("string"),
    "order":        Value("int64"),
})


# Numerical id columns dropped from the published references dataset —
# slug-based identifiers (`from_slug`, `to_slug`, `from_law_book_slug`,
# `to_law_book_slug`, `to_law_book_code`) are sufficient for downstream
# consumers and avoid leaking internal DB primary keys.
REFERENCES_DROP_COLUMNS = frozenset({"from_id", "to_id", "from_case_court_id"})


# Typed schema for the references dataset. 23 columns (the 26 source
# columns minus REFERENCES_DROP_COLUMNS). ``from_case_date`` carries an
# ISO ``YYYY-MM-DD`` string when set and an empty cell otherwise; we
# materialise it as ``date32`` so consumers can filter by date natively.
REFERENCES_FEATURES = Features({
    "from_case_court_chamber":         Value("string"),
    "from_case_court_city":            Value("string"),
    "from_case_court_jurisdiction":    Value("string"),
    "from_case_court_level_of_appeal": Value("string"),
    "from_case_court_name":            Value("string"),
    "from_case_court_state":           Value("string"),
    "from_case_date":                  Value("date32"),
    "from_case_file_number":           Value("string"),
    "from_case_review_status":         Value("string"),
    "from_case_source_name":           Value("string"),
    "from_case_type":                  Value("string"),
    "from_law_book_slug":              Value("string"),
    "from_slug":                       Value("string"),
    "from_type":                       Value("string"),
    "to_case_court_jurisdiction":      Value("string"),
    "to_case_court_level_of_appeal":   Value("string"),
    "to_case_court_name":              Value("string"),
    "to_law_book_code":                Value("string"),
    "to_law_book_slug":                Value("string"),
    "to_law_section":                  Value("string"),
    "to_law_title":                    Value("string"),
    "to_slug":                         Value("string"),
    "to_type":                         Value("string"),
})


def _jsonl_generator(file_path: str, skip: int = 0, limit: int = None):
    """Module-level generator over a JSONL (possibly .gz) file.

    Mirrors :py:meth:`ConvertDumpToHFCommand._stream_jsonl_data` but as a
    plain function so that :func:`datasets.Dataset.from_generator` can
    fingerprint and re-call it (the method/lambda variant fails to
    pickle when a live generator is in scope).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    is_gzipped = path.suffix.lower() == ".gz"
    open_func = gzip.open if is_gzipped else open
    mode = "rt" if is_gzipped else "r"

    entries_processed = 0
    with open_func(file_path, mode, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if i < skip:
                continue
            if limit and entries_processed >= limit:
                break
            try:
                yield json.loads(line)
                entries_processed += 1
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i + 1}: {e}")
                continue


def _detect_type_from_path(path: str) -> str | None:
    """Auto-detect resource type from the input file's basename.

    Returns 'cases' / 'laws' / 'references' if the first dot-separated
    token of the filename matches one of those; otherwise None. Handles
    plain names (cases.jsonl.gz), numeric-suffixed fixtures
    (cases.10.jsonl.gz), gzipped CSV (references.csv.gz), and absolute
    paths.
    """
    token = Path(path).name.split(".", 1)[0]
    return token if token in SUPPORTED_TYPES else None


class ConvertDumpToHFCommand(BaseCommand):
    """Command to convert data dumps to HuggingFace format and upload to Hub."""

    def __init__(self):
        super().__init__()
        self.ref_extractor = RefExtractor()

    def add_arguments(self, parser):
        """Add command-specific arguments."""
        parser.add_argument("input_file", help="Path to input dump file (JSONL.gz for cases/laws, CSV for references)")
        parser.add_argument("output", help="Output destination (HF repo ID, file path, or directory)")
        parser.add_argument(
            "--type",
            choices=list(SUPPORTED_TYPES),
            default=None,
            help=(
                "Resource type to convert. Auto-detected from the input "
                "filename (cases.*, laws.*, references.*) when omitted."
            ),
        )
        parser.add_argument(
            "--format",
            choices=["hf_hub", "jsonl", "parquet", "hf_disk"],
            default="hf_hub",
            help="Output format: hf_hub (HuggingFace Hub), jsonl, parquet, or hf_disk (HF format to disk)",
        )
        parser.add_argument("--skip", type=int, default=0, help="Skip the first N entries from the dataset")
        parser.add_argument("--limit", type=int, help="Limit to first N entries from the dataset")
        parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing (default: 1000)")
        parser.add_argument(
            "--private", action="store_true", help="Make the dataset repository private (hf_hub format only)"
        )
        parser.add_argument("--config-name", help="Configuration name for HF formats (required for hf_hub and hf_disk)")
        parser.add_argument("--split", default="train", help="Dataset split name for HF formats (default: train)")
        parser.add_argument("--no-process", action="store_true", help="Skip HTML to Markdown processing")
        parser.add_argument(
            "--num-proc",
            type=int,
            default=None,
            help="Number of processes to use for parallel processing (default: None, no parallelization)",
        )
        parser.add_argument(
            "--streaming",
            action="store_true",
            help="Enable streaming mode for JSONL loading (loads data line by line instead of all at once)",
        )

    def _load_jsonl_data(self, file_path: str, skip: int = 0, limit: int = None):
        """Load data from JSONL file (possibly gzipped)."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")
        if skip > 0:
            logger.info(f"Skipping first {skip} entries")

        # Determine if file is gzipped
        is_gzipped = path.suffix.lower() == ".gz"
        open_func = gzip.open if is_gzipped else open
        mode = "rt" if is_gzipped else "r"

        data = []
        entries_processed = 0
        with open_func(file_path, mode, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Skip entries if needed
                if i < skip:
                    continue

                # Apply limit after skipping
                if limit and entries_processed >= limit:
                    break

                try:
                    data.append(json.loads(line))
                    entries_processed += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {i + 1}: {e}")
                    continue

        logger.info(f"Loaded {len(data)} records (skipped {skip}, processed {entries_processed})")
        return data

    def _stream_jsonl_data(self, file_path: str, skip: int = 0, limit: int = None):
        """Stream data from JSONL file (possibly gzipped) using a generator."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Streaming data from {file_path}")
        if skip > 0:
            logger.info(f"Skipping first {skip} entries")

        # Determine if file is gzipped
        is_gzipped = path.suffix.lower() == ".gz"
        open_func = gzip.open if is_gzipped else open
        mode = "rt" if is_gzipped else "r"

        entries_processed = 0
        with open_func(file_path, mode, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Skip entries if needed
                if i < skip:
                    continue

                # Apply limit after skipping
                if limit and entries_processed >= limit:
                    break

                try:
                    yield json.loads(line)
                    entries_processed += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {i + 1}: {e}")
                    continue

        logger.info(f"Streamed {entries_processed} records (skipped {skip})")

    def process_case(self, example):
        """Process a single case entry.
        - Converting HTML content to Markdown.
        - Extract references

        Args:
            example: A dictionary containing case data with a 'content' field

        Returns:
            The example dictionary with an added 'markdown_content' and 'reference_markers' fields
        """
        if "content" in example and example["content"]:
            try:
                # Convert HTML to Markdown
                markdown_text = md(
                    example["content"],
                    heading_style="ATX",  # Use # for headings
                    bullets="-",  # Use - for bullet points
                    strip=["script", "style"],  # Remove script and style tags
                )
                example["markdown_content"] = markdown_text.strip()

                # Extract references
                _, markers = self.ref_extractor.extract(markdown_text)

                def _ref_to_dict(ref) -> dict:
                    """Convert a reference object to a JSON-serializable dictionary."""
                    return {
                        "ref_type": str(ref.ref_type if hasattr(ref, "ref_type") else "unknown"),
                        **{k: v for k, v in ref.__dict__.items() if k != "ref_type"},
                    }

                def _marker_to_dict(marker: RefMarker) -> dict:
                    """Convert a marker object to a JSON-serializable dictionary."""
                    marker_dict = {
                        **{
                            col: getattr(marker, col)
                            for col in ["end", "line", "start", "text"]
                            if hasattr(marker, col)
                        }
                    }

                    # Handle references within the marker
                    if hasattr(marker, "references") and marker.references:
                        marker_dict["references"] = [_ref_to_dict(ref) for ref in marker.references]
                    else:
                        marker_dict["references"] = []

                    return marker_dict

                # example["content_with_markers"] = content_with_markers
                reference_markers = [_marker_to_dict(marker) for marker in markers]
                reference_markers = json.dumps(reference_markers)  # stringify for PyArrow
                example["reference_markers"] = reference_markers

            except Exception as e:
                logger.error(f"Failed to process case: {e}")
                example["markdown_content"] = example.get("content", "")
                example["reference_markers"] = "[]"
        else:
            logger.error("Content field missing")
            example["markdown_content"] = ""
            example["reference_markers"] = "[]"

        return example

    def _load_csv_data(self, file_path: str, skip: int = 0, limit: int = None, chunksize: int = 100_000):
        """Load the references CSV with a typed HuggingFace ``Features`` schema.

        Streams the CSV in pandas chunks (memory ceiling stays in the
        low-GB range for the full 7.5M-row file) and:

        - drops the numerical id columns via ``usecols`` at read time;
        - parses ``from_case_date`` as a date (empty cells -> NaT -> null);
        - reads all other columns as the nullable pandas string dtype so
          empty cells become ``<NA>`` -> null.

        Honours ``skip`` and ``limit`` (rows-after-skip), matching the
        JSONL loader's semantics. ``compression='infer'`` handles both
        ``references.csv`` and ``references.csv.gz``.

        Returns a :class:`datasets.Dataset` built with
        :data:`REFERENCES_FEATURES`, or ``None`` if no rows pass through.
        """
        import pandas as pd

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Loading CSV from {file_path}")
        if skip > 0:
            logger.info(f"Skipping first {skip} entries")

        # All remaining (non-dropped, non-date) columns are read as the
        # nullable pandas string dtype so empty cells become <NA>.
        keep_cols = set(REFERENCES_FEATURES) - {"from_case_date"}
        string_dtypes = {col: "string" for col in keep_cols}

        chunks = []
        rows_seen = 0  # data rows seen across chunks (header not counted by read_csv)
        rows_kept = 0
        for chunk in pd.read_csv(
            file_path,
            usecols=lambda c: c not in REFERENCES_DROP_COLUMNS,
            dtype=string_dtypes,
            parse_dates=["from_case_date"],
            na_values=[""],
            chunksize=chunksize,
            compression="infer",
        ):
            # Apply skip: drop rows from the front of this chunk if we
            # haven't yet skipped enough.
            if skip and rows_seen < skip:
                drop_n = min(skip - rows_seen, len(chunk))
                chunk = chunk.iloc[drop_n:]
                rows_seen += drop_n
            rows_seen += len(chunk)

            # Apply limit.
            if limit is not None:
                remaining = limit - rows_kept
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]

            if len(chunk) == 0:
                continue

            chunks.append(chunk)
            rows_kept += len(chunk)
            if limit is not None and rows_kept >= limit:
                break

        if not chunks:
            return None

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df)} rows (skipped {skip})")
        return Dataset.from_pandas(df, features=REFERENCES_FEATURES, preserve_index=False)

    def process_law(self, example):
        """Process a single law entry.
        - Convert HTML 'content' to Markdown into 'markdown_content'.

        Unlike :py:meth:`process_case`, this intentionally does NOT extract
        references — citations in laws are reconstructable from the
        separate ``references.csv`` artifact, and keeping the law schema
        free of marker JSON keeps the published dataset cleaner.

        Args:
            example: A dict for one law row, expected to contain ``content``.

        Returns:
            The example with an added ``markdown_content`` field.
        """
        if example.get("content"):
            try:
                markdown_text = md(
                    example["content"],
                    heading_style="ATX",
                    bullets="-",
                    strip=["script", "style"],
                )
                example["markdown_content"] = markdown_text.strip()
            except Exception as e:
                logger.error(f"Failed to process law: {e}")
                example["markdown_content"] = example.get("content", "")
        else:
            logger.error("Content field missing")
            example["markdown_content"] = ""
        return example

    def _save_to_hub(self, dataset, output: str, private: bool = False, config_name: str = None, split: str = "train"):
        """Save dataset to HuggingFace Hub."""
        logger.info(f"Pushing dataset to HuggingFace Hub: {output}")
        logger.info(f"Config: {config_name}, Split: {split}")
        dataset.push_to_hub(output, config_name=config_name, split=split, private=private)
        logger.info("Dataset upload completed successfully!")
        logger.info(f"Successfully uploaded {len(dataset)} records to {output} (config: {config_name}, split: {split})")

    def _save_to_jsonl(self, dataset, output: str):
        """Save dataset to JSONL format."""
        logger.info(f"Saving dataset to JSONL: {output}")

        # Use HF datasets to_json for remote filesystem support
        dataset.to_json(output)

        logger.info("JSONL export completed successfully!")
        logger.info(f"Successfully saved {len(dataset)} records to {output}")

    def _save_to_parquet(self, dataset, output: str):
        """Save dataset to Parquet format."""
        logger.info(f"Saving dataset to Parquet: {output}")

        # Use HF datasets to_parquet for remote filesystem support
        dataset.to_parquet(output)

        logger.info("Parquet export completed successfully!")
        logger.info(f"Successfully saved {len(dataset)} records to {output}")

    def _save_to_disk(self, dataset, output: str, config_name: str = None, split: str = "train"):
        """Save dataset to disk in HuggingFace format."""
        logger.info(f"Saving dataset to disk: {output}")
        logger.info(f"Config: {config_name}, Split: {split}")

        # For save_to_disk, we need to handle the split structure manually
        from datasets import DatasetDict

        if config_name:
            # Create nested structure: config -> split -> dataset
            dataset_dict = DatasetDict({split: dataset})
            # Save with config structure
            full_path = f"{output}/{config_name}"
            dataset_dict.save_to_disk(full_path)
        else:
            # Simple split structure
            dataset_dict = DatasetDict({split: dataset})
            dataset_dict.save_to_disk(output)

        logger.info("Dataset save to disk completed successfully!")
        logger.info(f"Successfully saved {len(dataset)} records to {output} (config: {config_name}, split: {split})")

    def _save_dataset(self, dataset, args):
        """Save dataset based on the specified format."""
        output_format = args.format.lower()

        # Validate required config_name for HF formats
        if output_format in ("hf_hub", "hf_disk") and not args.config_name:
            raise ValueError(f"--config-name is required for {output_format} format")

        if output_format == "hf_hub":
            self._save_to_hub(dataset, args.output, args.private, args.config_name, args.split)
        elif output_format == "jsonl":
            self._save_to_jsonl(dataset, args.output)
        elif output_format == "parquet":
            self._save_to_parquet(dataset, args.output)
        elif output_format == "hf_disk":
            self._save_to_disk(dataset, args.output, args.config_name, args.split)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def handle(self, args):
        """Execute the convert_dump_to_hf command."""
        try:
            resource_type = args.type or _detect_type_from_path(args.input_file)
            if resource_type is None:
                raise ValueError(
                    "Could not auto-detect resource type from filename. "
                    f"Pass --type with one of: {', '.join(SUPPORTED_TYPES)}."
                )

            logger.info(f"Converting dump from {args.input_file}")
            logger.info(f"Resource type: {resource_type}")
            logger.info(f"Output format: {args.format}")
            logger.info(f"Output destination: {args.output}")
            if args.skip:
                logger.info(f"Skipping first {args.skip} entries")
            if args.limit:
                logger.info(f"Limiting to first {args.limit} entries after skip")

            if resource_type == "references":
                dataset = self._build_dataset_references(args)
                if dataset is None:
                    return
            else:
                # JSONL: cases or laws
                dataset = self._build_dataset_jsonl(args, resource_type)
                if dataset is None:
                    return

            # Save dataset in the specified format
            self._save_dataset(dataset, args)

        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            raise

    def _build_dataset_jsonl(self, args, resource_type):
        """Load JSONL (cases/laws), optionally process, return a Dataset."""
        if args.streaming:
            logger.info("Using streaming mode for JSONL loading")
            logger.info("Creating HuggingFace dataset from stream")
            # ``Dataset.from_generator`` fingerprints + pickles its callable
            # for caching. A lambda closing over a live generator can't be
            # pickled; pass an unbound module-level wrapper and the
            # per-call kwargs via ``gen_kwargs`` instead.
            #
            # ``features=`` is required for nested-struct columns whose
            # first-batch values are nullable (Arrow otherwise infers
            # ``null`` and later non-null values fail to cast).
            features = (
                CASES_FEATURES if resource_type == "cases"
                else LAWS_FEATURES if resource_type == "laws"
                else None
            )
            dataset = Dataset.from_generator(
                _jsonl_generator,
                gen_kwargs={
                    "file_path": args.input_file,
                    "skip": args.skip,
                    "limit": args.limit,
                },
                features=features,
            )
        else:
            data = self._load_jsonl_data(args.input_file, args.skip, args.limit)
            if not data:
                logger.error("No valid data found in input file")
                return None
            logger.info("Creating HuggingFace dataset from list")
            dataset = Dataset.from_list(data)

        if not args.no_process:
            processor = self.process_case if resource_type == "cases" else self.process_law
            desc = f"Processing {resource_type}"
            if args.num_proc:
                logger.info(f"{desc} with {args.num_proc} processes")
            else:
                logger.info(f"{desc} (single process)")
            dataset = dataset.map(processor, batched=False, desc=desc, num_proc=args.num_proc)
            logger.info("Processing completed")

        return dataset

    def _build_dataset_references(self, args):
        """Load references CSV with typed schema, return a Dataset."""
        if args.streaming:
            logger.warning(
                "--streaming is ignored for --type references "
                "(CSV is loaded in chunks but materialised in memory)."
            )
        if args.no_process:
            logger.warning("--no-process is ignored for --type references (no processing step exists).")
        # _load_csv_data is added in task #3
        dataset = self._load_csv_data(args.input_file, args.skip, args.limit)
        if dataset is None:
            logger.error("No valid data found in input file")
        return dataset
