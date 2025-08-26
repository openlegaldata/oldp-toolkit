"""Base command class for OLDP Toolkit commands."""

import logging
import sys
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset to default color

    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and self._supports_color()

    def _supports_color(self):
        """Check if the terminal supports colors."""
        # Check if output is a TTY (terminal) and not redirected
        if not hasattr(sys.stdout, "isatty"):
            return False
        if not sys.stdout.isatty():
            return False
        # Additional check for Windows
        if sys.platform == "win32":
            try:
                import os

                return os.getenv("TERM") is not None or "ANSI" in os.getenv("TERM", "")
            except Exception:
                return False
        return True

    def format(self, record):
        if self.use_colors and record.levelname in self.COLORS:
            # Color the level name
            level_color = self.COLORS[record.levelname]
            record.levelname = f"{level_color}{record.levelname}{self.RESET}"

        return super().format(record)


class BaseCommand(ABC):
    """Base class for all CLI commands."""

    @abstractmethod
    def add_arguments(self, parser):
        """Add command-specific arguments to the parser."""
        pass

    @abstractmethod
    def handle(self, args):
        """Execute the command logic."""
        pass

    def setup_logging(self, debug=False):
        """Configure logging based on debug flag."""
        level = logging.DEBUG if debug else logging.INFO

        # Create colored formatter
        formatter = ColoredFormatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create and configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)
