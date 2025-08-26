"""Tests for BaseCommand class."""

from unittest.mock import Mock

import pytest

from oldp_toolkit.commands.base import BaseCommand, ColoredFormatter


class MockCommand(BaseCommand):
    """Mock command for testing."""

    def add_arguments(self, parser):
        parser.add_argument("--test", help="Test argument")

    def handle(self, args):
        return "handled"


def test_base_command_abstract():
    """Test that BaseCommand cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseCommand()


def test_mock_command_instantiation():
    """Test that concrete command can be instantiated."""
    command = MockCommand()
    assert command is not None


def test_setup_logging_default():
    """Test setup_logging with default (non-debug) mode."""
    command = MockCommand()
    command.setup_logging(debug=False)
    # Logger should be configured at INFO level


def test_setup_logging_debug():
    """Test setup_logging with debug mode."""
    command = MockCommand()
    command.setup_logging(debug=True)
    # Logger should be configured at DEBUG level


def test_add_arguments():
    """Test add_arguments method."""
    command = MockCommand()
    parser = Mock()
    command.add_arguments(parser)
    parser.add_argument.assert_called_once_with("--test", help="Test argument")


def test_handle():
    """Test handle method."""
    command = MockCommand()
    args = Mock()
    result = command.handle(args)
    assert result == "handled"


def test_colored_formatter_with_colors():
    """Test ColoredFormatter with colors enabled."""
    import logging

    formatter = ColoredFormatter(fmt="%(levelname)s: %(message)s", use_colors=True)
    formatter.use_colors = True  # Force colors for testing

    # Create a log record
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="Test message", args=(), exc_info=None
    )

    formatted = formatter.format(record)

    # Check that ANSI codes are present
    assert "\033[32m" in formatted  # Green color for INFO
    assert "\033[0m" in formatted  # Reset code


def test_colored_formatter_without_colors():
    """Test ColoredFormatter with colors disabled."""
    import logging

    formatter = ColoredFormatter(use_colors=False)

    # Create a log record
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="Test message", args=(), exc_info=None
    )

    formatted = formatter.format(record)

    # Check that no ANSI codes are present
    assert "\033[" not in formatted


def test_colored_formatter_supports_color():
    """Test the _supports_color method."""
    formatter = ColoredFormatter()

    # This will return True or False depending on the test environment
    # Just ensure the method doesn't crash
    result = formatter._supports_color()
    assert isinstance(result, bool)
