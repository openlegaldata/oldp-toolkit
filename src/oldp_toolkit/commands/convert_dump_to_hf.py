"""Convert dump to HuggingFace format command."""

import gzip
import json
import logging
from pathlib import Path

from datasets import Dataset
from markdownify import markdownify as md
from refex.extractor import RefExtractor, RefMarker

from oldp_toolkit.commands.base import BaseCommand

logger = logging.getLogger(__name__)


class ConvertDumpToHFCommand(BaseCommand):
    """Command to convert data dumps to HuggingFace format and upload to Hub."""

    def __init__(self):
        super().__init__()
        self.ref_extractor = RefExtractor()

    def add_arguments(self, parser):
        """Add command-specific arguments."""
        parser.add_argument("input_file", help="Path to input JSONL dump file (can be gzipped)")
        parser.add_argument("output", help="Output destination (HF repo ID, file path, or directory)")
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
            logger.info(f"Converting dump from {args.input_file}")
            logger.info(f"Output format: {args.format}")
            logger.info(f"Output destination: {args.output}")
            if args.skip:
                logger.info(f"Skipping first {args.skip} entries")
            if args.limit:
                logger.info(f"Limiting to first {args.limit} entries after skip")

            # Load data from JSONL file
            data = self._load_jsonl_data(args.input_file, args.skip, args.limit)

            if not data:
                logger.error("No valid data found in input file")
                return

            # Create HuggingFace dataset
            logger.info("Creating HuggingFace dataset")
            dataset = Dataset.from_list(data)

            # Process cases unless disabled
            if not args.no_process:
                logger.info("Processing cases")
                dataset = dataset.map(self.process_case, batched=False, desc="Processing cases")
                logger.info("Processing completed")

            # Save dataset in the specified format
            self._save_dataset(dataset, args)

        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            raise
