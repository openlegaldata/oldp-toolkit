"""Main CLI entrypoint for OLDP Toolkit."""

import argparse
import sys

from oldp_toolkit.commands.convert_dump_to_hf import ConvertDumpToHFCommand


def get_commands():
    """Return a dictionary of available commands."""
    return {
        "convert_dump_to_hf": ConvertDumpToHFCommand,
    }


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="oldpt", description="OLDP Toolkit - Command-line tools for Open Legal Data Platform"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    commands = get_commands()
    command_instances = {}

    for name, command_class in commands.items():
        command_instances[name] = command_class()
        subparser = subparsers.add_parser(name, help=f"Run {name} command")
        command_instances[name].add_arguments(subparser)

    args = parser.parse_args()

    if args.command not in command_instances:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    command = command_instances[args.command]
    command.setup_logging(debug=args.debug)

    try:
        command.handle(args)
        return 0
    except Exception as e:
        if args.debug:
            raise
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
