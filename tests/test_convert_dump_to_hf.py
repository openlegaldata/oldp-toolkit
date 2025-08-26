"""Tests for ConvertDumpToHFCommand."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oldp_toolkit.commands.convert_dump_to_hf import ConvertDumpToHFCommand


def test_convert_dump_command_instantiation():
    """Test that ConvertDumpToHFCommand can be instantiated."""
    command = ConvertDumpToHFCommand()
    assert command is not None


def test_add_arguments():
    """Test add_arguments method adds required arguments."""
    command = ConvertDumpToHFCommand()
    parser = Mock()

    command.add_arguments(parser)

    # Check that required arguments are added
    calls = parser.add_argument.call_args_list
    call_args = [call[0] for call in calls]

    assert ("input_file",) in call_args
    assert ("output",) in call_args
    assert ("--format",) in call_args
    assert ("--skip",) in call_args
    assert ("--limit",) in call_args
    assert ("--batch-size",) in call_args
    assert ("--private",) in call_args
    assert ("--config-name",) in call_args
    assert ("--split",) in call_args
    assert ("--no-process",) in call_args


def test_load_jsonl_data():
    """Test _load_jsonl_data method."""
    command = ConvertDumpToHFCommand()

    # Create temporary JSONL file
    test_data = [
        {"id": 1, "text": "First document"},
        {"id": 2, "text": "Second document"},
        {"id": 3, "text": "Third document"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        # Test loading all data
        result = command._load_jsonl_data(temp_path)
        assert len(result) == 3
        assert result[0]["id"] == 1
        assert result[2]["text"] == "Third document"

        # Test loading with limit
        result_limited = command._load_jsonl_data(temp_path, limit=2)
        assert len(result_limited) == 2
        assert result_limited[1]["id"] == 2

    finally:
        Path(temp_path).unlink()


def test_load_jsonl_data_with_skip():
    """Test _load_jsonl_data method with skip parameter."""
    command = ConvertDumpToHFCommand()

    # Create temporary JSONL file with 5 items
    test_data = [
        {"id": 1, "text": "First document"},
        {"id": 2, "text": "Second document"},
        {"id": 3, "text": "Third document"},
        {"id": 4, "text": "Fourth document"},
        {"id": 5, "text": "Fifth document"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        # Test skipping first 2 entries
        result = command._load_jsonl_data(temp_path, skip=2)
        assert len(result) == 3
        assert result[0]["id"] == 3  # Should start from 3rd entry
        assert result[2]["id"] == 5  # Should end with 5th entry

        # Test skip with limit
        result_skip_limit = command._load_jsonl_data(temp_path, skip=1, limit=2)
        assert len(result_skip_limit) == 2
        assert result_skip_limit[0]["id"] == 2  # Should start from 2nd entry
        assert result_skip_limit[1]["id"] == 3  # Should end with 3rd entry

        # Test skip beyond available data
        result_skip_all = command._load_jsonl_data(temp_path, skip=10)
        assert len(result_skip_all) == 0

    finally:
        Path(temp_path).unlink()


def test_load_jsonl_data_file_not_found():
    """Test _load_jsonl_data with non-existent file."""
    command = ConvertDumpToHFCommand()

    with pytest.raises(FileNotFoundError):
        command._load_jsonl_data("non_existent_file.jsonl")


def test_process_case_with_html():
    """Test process_case function with HTML content."""
    command = ConvertDumpToHFCommand()

    example = {"id": 1, "content": "<h1>Title</h1><p>This is a <strong>paragraph</strong> with <em>emphasis</em>.</p>"}

    result = command.process_case(example)

    assert "markdown_content" in result
    assert "# Title" in result["markdown_content"]
    assert "**paragraph**" in result["markdown_content"]
    assert "*emphasis*" in result["markdown_content"]
    assert "reference_markers" in result
    assert isinstance(result["reference_markers"], str)  # Should be JSON string
    assert result["id"] == 1  # Original fields preserved


def test_process_case_without_content():
    """Test process_case function without content field."""
    command = ConvertDumpToHFCommand()

    example = {"id": 1, "title": "Test Case"}

    result = command.process_case(example)

    assert "markdown_content" in result
    assert result["markdown_content"] == ""
    assert "reference_markers" in result
    assert result["reference_markers"] == "[]"
    assert result["id"] == 1


def test_process_case_with_empty_content():
    """Test process_case function with empty content."""
    command = ConvertDumpToHFCommand()

    example = {"id": 1, "content": ""}

    result = command.process_case(example)

    assert "markdown_content" in result
    assert result["markdown_content"] == ""
    assert "reference_markers" in result
    assert result["reference_markers"] == "[]"


def test_process_case_with_malformed_html():
    """Test process_case function with malformed HTML."""
    command = ConvertDumpToHFCommand()

    example = {"id": 1, "content": "<h1>Title<p>Unclosed tags"}

    result = command.process_case(example)

    # Should still process despite malformed HTML
    assert "markdown_content" in result
    assert len(result["markdown_content"]) > 0


@patch("oldp_toolkit.commands.convert_dump_to_hf.Dataset")
def test_handle_success(mock_dataset_class):
    """Test successful handle execution."""
    command = ConvertDumpToHFCommand()

    # Create test data
    test_data = [{"id": 1, "text": "Test document"}]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(test_data[0]) + "\n")
        temp_path = f.name

    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=1)  # Add len() method
    mock_dataset_class.from_list.return_value = mock_dataset

    # Mock args
    args = Mock()
    args.input_file = temp_path
    args.output = "test/dataset"
    args.format = "hf_hub"
    args.skip = 0
    args.limit = None
    args.private = False
    args.config_name = "default"
    args.split = "train"
    args.no_process = True  # Skip processing for this test
    args.batch_size = 1000

    try:
        # Execute handle
        command.handle(args)

        # Verify dataset creation and upload
        mock_dataset_class.from_list.assert_called_once_with(test_data)
        mock_dataset.push_to_hub.assert_called_once_with(
            "test/dataset", config_name="default", split="train", private=False
        )

    finally:
        Path(temp_path).unlink()


@patch("oldp_toolkit.commands.convert_dump_to_hf.Dataset")
def test_handle_no_data(mock_dataset_class):
    """Test handle with empty input file."""
    command = ConvertDumpToHFCommand()

    # Create empty file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        temp_path = f.name

    args = Mock()
    args.input_file = temp_path
    args.output = "test/dataset"
    args.format = "hf_hub"
    args.skip = 0
    args.limit = None
    args.no_process = True

    try:
        # Execute handle
        command.handle(args)

        # Should not create dataset if no data
        mock_dataset_class.from_list.assert_not_called()

    finally:
        Path(temp_path).unlink()


@patch("oldp_toolkit.commands.convert_dump_to_hf.Dataset")
def test_handle_with_processing(mock_dataset_class):
    """Test handle with HTML to Markdown processing enabled."""
    command = ConvertDumpToHFCommand()

    # Create test data with HTML content
    test_data = [{"id": 1, "content": "<h1>Title</h1><p>Content</p>"}]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(test_data[0]) + "\n")
        temp_path = f.name

    # Mock dataset and its map method
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=1)  # Add len() method
    mock_processed_dataset = Mock()
    mock_processed_dataset.__len__ = Mock(return_value=1)  # Add len() method
    mock_dataset.map.return_value = mock_processed_dataset
    mock_dataset_class.from_list.return_value = mock_dataset

    # Mock args with processing enabled
    args = Mock()
    args.input_file = temp_path
    args.output = "test/dataset"
    args.format = "hf_hub"
    args.skip = 0
    args.limit = None
    args.private = False
    args.config_name = "default"
    args.split = "train"
    args.no_process = False  # Enable processing
    args.batch_size = 1000

    try:
        # Execute handle
        command.handle(args)

        # Verify dataset creation
        mock_dataset_class.from_list.assert_called_once_with(test_data)

        # Verify processing was applied
        mock_dataset.map.assert_called_once()
        call_args = mock_dataset.map.call_args
        assert call_args[0][0] == command.process_case
        assert not call_args[1]["batched"]
        assert "desc" in call_args[1]

        # Verify upload used processed dataset
        mock_processed_dataset.push_to_hub.assert_called_once_with(
            "test/dataset", config_name="default", split="train", private=False
        )

    finally:
        Path(temp_path).unlink()


def test_save_to_hub():
    """Test _save_to_hub method."""
    command = ConvertDumpToHFCommand()

    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)

    command._save_to_hub(mock_dataset, "test/dataset", private=True, config_name="default", split="train")

    mock_dataset.push_to_hub.assert_called_once_with("test/dataset", config_name="default", split="train", private=True)


def test_save_to_jsonl():
    """Test _save_to_jsonl method."""
    command = ConvertDumpToHFCommand()

    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)

    command._save_to_jsonl(mock_dataset, "output.jsonl")

    mock_dataset.to_json.assert_called_once_with("output.jsonl")


def test_save_to_parquet():
    """Test _save_to_parquet method."""
    command = ConvertDumpToHFCommand()

    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)

    command._save_to_parquet(mock_dataset, "output.parquet")

    mock_dataset.to_parquet.assert_called_once_with("output.parquet")


@patch("datasets.DatasetDict")
def test_save_to_disk(mock_dataset_dict_class):
    """Test _save_to_disk method."""
    command = ConvertDumpToHFCommand()

    # Mock dataset and DatasetDict
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)
    mock_dataset_dict = Mock()
    mock_dataset_dict_class.return_value = mock_dataset_dict

    command._save_to_disk(mock_dataset, "output_dir", config_name="default", split="train")

    # Check that DatasetDict was created with the dataset
    mock_dataset_dict_class.assert_called_once_with({"train": mock_dataset})
    # Check that save_to_disk was called with config path
    mock_dataset_dict.save_to_disk.assert_called_once_with("output_dir/default")


def test_save_dataset_hf_hub_format():
    """Test _save_dataset with hf_hub format."""
    command = ConvertDumpToHFCommand()

    mock_dataset = Mock()
    args = Mock()
    args.format = "hf_hub"
    args.output = "test/dataset"
    args.private = False
    args.config_name = "default"
    args.split = "train"
    args.config_name = "default"
    args.split = "train"

    with patch.object(command, "_save_to_hub") as mock_save:
        command._save_dataset(mock_dataset, args)
        mock_save.assert_called_once_with(mock_dataset, "test/dataset", False, "default", "train")


def test_save_dataset_jsonl_format():
    """Test _save_dataset with jsonl format."""
    command = ConvertDumpToHFCommand()

    mock_dataset = Mock()
    args = Mock()
    args.format = "jsonl"
    args.output = "output.jsonl"

    with patch.object(command, "_save_to_jsonl") as mock_save:
        command._save_dataset(mock_dataset, args)
        mock_save.assert_called_once_with(mock_dataset, "output.jsonl")


def test_save_dataset_hf_disk_format():
    """Test _save_dataset with hf_disk format."""
    command = ConvertDumpToHFCommand()

    mock_dataset = Mock()
    args = Mock()
    args.format = "hf_disk"
    args.output = "output_dir"
    args.config_name = "default"
    args.split = "train"

    with patch.object(command, "_save_to_disk") as mock_save:
        command._save_dataset(mock_dataset, args)
        mock_save.assert_called_once_with(mock_dataset, "output_dir", "default", "train")


def test_save_dataset_unsupported_format():
    """Test _save_dataset with unsupported format."""
    command = ConvertDumpToHFCommand()

    mock_dataset = Mock()
    args = Mock()
    args.format = "unsupported"

    with pytest.raises(ValueError, match="Unsupported output format"):
        command._save_dataset(mock_dataset, args)


def test_save_dataset_missing_config_name_hf_hub():
    """Test _save_dataset with hf_hub format but missing config_name."""
    command = ConvertDumpToHFCommand()

    mock_dataset = Mock()
    args = Mock()
    args.format = "hf_hub"
    args.config_name = None

    with pytest.raises(ValueError, match="--config-name is required for hf_hub format"):
        command._save_dataset(mock_dataset, args)


def test_save_dataset_missing_config_name_hf_disk():
    """Test _save_dataset with hf_disk format but missing config_name."""
    command = ConvertDumpToHFCommand()

    mock_dataset = Mock()
    args = Mock()
    args.format = "hf_disk"
    args.config_name = None

    with pytest.raises(ValueError, match="--config-name is required for hf_disk format"):
        command._save_dataset(mock_dataset, args)


def test_full_processing_with_fixture_data():
    """Integration test using the fixture file with 10 legal case entries."""
    import json
    import tempfile
    from pathlib import Path

    from datasets import Dataset

    command = ConvertDumpToHFCommand()

    # Get the fixture file path
    fixture_path = Path(__file__).parent / "fixtures" / "dumps" / "cases.10.jsonl.gz"

    # Verify fixture exists
    assert fixture_path.exists(), f"Fixture file not found at {fixture_path}"

    # Test data loading
    data = command._load_jsonl_data(str(fixture_path))

    # Verify we loaded exactly 10 entries
    assert len(data) == 10, f"Expected 10 entries, got {len(data)}"

    # Verify data structure - each entry should have expected fields
    for i, entry in enumerate(data):
        assert "id" in entry, f"Entry {i} missing 'id' field"
        assert "content" in entry, f"Entry {i} missing 'content' field"
        assert isinstance(entry["content"], str), f"Entry {i} 'content' should be string"
        assert len(entry["content"]) > 0, f"Entry {i} has empty content"

    # Test with different skip/limit parameters
    data_skip_2 = command._load_jsonl_data(str(fixture_path), skip=2)
    assert len(data_skip_2) == 8, "Skip 2 should return 8 entries"

    data_limit_5 = command._load_jsonl_data(str(fixture_path), limit=5)
    assert len(data_limit_5) == 5, "Limit 5 should return 5 entries"

    data_skip_2_limit_3 = command._load_jsonl_data(str(fixture_path), skip=2, limit=3)
    assert len(data_skip_2_limit_3) == 3, "Skip 2 limit 3 should return 3 entries"

    # Test processing a subset of data
    test_data = data[:3]  # Process first 3 entries

    # Create HuggingFace dataset
    dataset = Dataset.from_list(test_data)

    # Test processing cases - this tests the full HTML to Markdown conversion
    # and reference extraction pipeline
    processed_dataset = dataset.map(command.process_case, batched=False)

    # Verify processing results
    processed_data = processed_dataset.to_list()

    for i, processed_entry in enumerate(processed_data):
        # Original fields should be preserved
        assert processed_entry["id"] == test_data[i]["id"]
        assert processed_entry["content"] == test_data[i]["content"]

        # New fields should be added
        assert "markdown_content" in processed_entry, f"Entry {i} missing markdown_content"
        assert "reference_markers" in processed_entry, f"Entry {i} missing reference_markers"

        # markdown_content should be different from original HTML content
        assert processed_entry["markdown_content"] != processed_entry["content"]
        assert len(processed_entry["markdown_content"]) > 0, f"Entry {i} has empty markdown_content"

        # reference_markers should be a JSON string
        assert isinstance(processed_entry["reference_markers"], str)
        # Should be valid JSON
        markers = json.loads(processed_entry["reference_markers"])
        assert isinstance(markers, list), "reference_markers should be a list"

    # Test HTML to Markdown conversion quality on first entry
    first_entry = processed_data[0]
    markdown_content = first_entry["markdown_content"]

    # Should convert HTML headings to Markdown
    assert "# " in markdown_content or "## " in markdown_content, "Should contain Markdown headings"

    # Should remove HTML tags
    assert "<h" not in markdown_content, "Should not contain HTML heading tags"
    assert "<p>" not in markdown_content, "Should not contain HTML paragraph tags"

    # Test save dataset functionality with mock
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test saving to different formats
        mock_args = Mock()
        mock_args.format = "jsonl"
        mock_args.output = str(Path(temp_dir) / "output.jsonl")

        # Mock the dataset methods
        with patch.object(processed_dataset, "to_json") as mock_to_json:
            command._save_dataset(processed_dataset, mock_args)
            mock_to_json.assert_called_once_with(mock_args.output)
