"""Tests for CLI functionality."""

from unittest.mock import patch

import pytest

from oldp_toolkit.cli import get_commands, main


def test_get_commands():
    """Test that get_commands returns available commands."""
    commands = get_commands()
    assert "convert_dump_to_hf" in commands
    assert len(commands) >= 1


def test_main_no_args():
    """Test main function with no arguments shows help."""
    with patch("sys.argv", ["oldpt"]), pytest.raises(SystemExit):
        main()


def test_main_help():
    """Test main function with help argument."""
    with patch("sys.argv", ["oldpt", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


def test_main_unknown_command():
    """Test main function with unknown command."""
    with patch("sys.argv", ["oldpt", "unknown_command"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0
