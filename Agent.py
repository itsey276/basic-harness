"""Agent originally authored in Agent.ipynb.

Install dependencies with: pip install tavily-python openai requests beautifulsoup4
Set TAVILY_API_KEY and UF_API_KEY in the environment before running.
"""

import argparse
import json
import os
import subprocess

from openai import OpenAI
from tavily import TavilyClient


def tavily_search(query):
    """Perform a web search for query using Tavily."""
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = tavily_client.search(query)
    return str(response)


def fetch_page(url: str) -> str:
    """Fetch the text content of a web page at a URL."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()


def run_bash_command(command: str) -> str:
    """Execute a bash command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        if not output:
            output = "Command executed successfully with no output."
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"


def read_text_file(file_path: str) -> str:
    """Read a text file and return its contents."""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_text_file(file_path: str, content: str) -> str:
    """Write content to a text file and report the result."""
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        return f"File '{file_path}' has been written successfully."
    except Exception as e:
        return f"Error writing file: {str(e)}"


def edit_text_file(file_path: str, old_text: str, new_text: str) -> str:
    """Replace one uniquely matching text snippet in a file."""
    if not old_text:
        return "Error editing file: old_text must not be empty."

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        matches = content.count(old_text)
        if matches == 0:
            return f"Error editing file: The specified text was not found in '{file_path}'."
        if matches > 1:
            return (
                f"Error editing file: The specified text occurs {matches} times in "
                f"'{file_path}'; provide a more specific snippet."
            )

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content.replace(old_text, new_text, 1))
        return f"File '{file_path}' has been edited successfully."
    except FileNotFoundError:
        return f"Error editing file: File '{file_path}' not found."
    except Exception as e:
        return f"Error editing file: {str(e)}"


edit_text_file_tool = {
    "type": "function",
    "function": {
        "name": "edit_text_file",
        "description": "Replace a uniquely matching text snippet in a text file. The old_text must occur exactly once.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the text file to edit.",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find; it must occur exactly once.",
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace the matching snippet with.",
                },
            },
            "required": ["file_path", "old_text", "new_text"],
        },
    },
}


write_text_file_tool = {
    "type": "function",
    "function": {
        "name": "write_text_file",
        "description": "Write content to a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the text file.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        },
    },
}

read_text_file_tool = {
    "type": "function",
    "function": {
        "name": "read_text_file",
        "description": "Read the contents of a text file and return it as a string.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the text file.",
                }
            },
            "required": ["file_path"],
        },
    },
}

run_bash_command_tool = {
    "type": "function",
    "function": {
        "name": "run_bash_command",
        "description": "Execute a bash command and return the output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                }
            },
            "required": ["command"],
        },
    },
}

fetch_page_tool = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": "Fetches the text content of a web page at a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the web page to fetch.",
                }
            },
            "required": ["url"],
        },
    },
}

tavily_search_tool = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": "Perform a web search for a query using Tavily.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                }
            },
            "required": ["query"],
        },
    },
}

client = OpenAI(
    api_key=os.environ["UF_API_KEY"],
    base_url="https://api.ai.it.ufl.edu/v1",
)


def get_completion(messages, model):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[
            edit_text_file_tool,
            write_text_file_tool,
            tavily_search_tool,
            fetch_page_tool,
            run_bash_command_tool,
            read_text_file_tool,
        ],
        tool_choice="auto",
    )
    return response.choices[0].message


tool_map = {
    "tavily_search": tavily_search,
    "fetch_page": fetch_page,
    "run_bash_command": run_bash_command,
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "edit_text_file": edit_text_file,
}


def message_field(message, field):
    """Read a field from either a dict or an OpenAI message object."""
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def message_to_dict(message):
    """Convert an OpenAI message or dict to a JSON-serializable dict."""
    if isinstance(message, dict):
        return message
    return message.model_dump(exclude_none=True)


def execute_tool_calls(messages):
    """Execute tool calls from the last assistant message."""
    message = messages[-1]
    tool_calls = message_field(message, "tool_calls") or []

    for tool_call in tool_calls:
        function = message_field(tool_call, "function")
        function_name = message_field(function, "name")
        function_args = json.loads(message_field(function, "arguments"))
        tool_call_id = message_field(tool_call, "id")

        print(f"Executing tool call {tool_call_id} with function: {function_name}")
        print(f"Function arguments: {function_args}")
        print("=" * 30)

        try:
            function = tool_map.get(function_name)
            if function:
                response = function(**function_args)
                messages.append(
                    {"tool_call_id": tool_call_id, "role": "tool", "content": response}
                )
            else:
                messages.append(
                    {"tool_call_id": tool_call_id, "role": "tool", "content": "Invalid tool"}
                )
        except Exception as e:
            messages.append(
                {
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "content": f"Tool Error: {e}",
                }
            )

    return messages


def main():
    parser = argparse.ArgumentParser(description="Run the research agent with a user query.")
    parser.add_argument("query", nargs="?", help="The question or task for the agent.")
    parser.add_argument("--save", metavar="FILENAME", help="Save the full message context to a JSON file.")
    parser.add_argument("--load", metavar="FILENAME", help="Load a message context from a JSON file.")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum number of agent iterations (default: 10).",
    )
    args = parser.parse_args()

    if args.load:
        with open(args.load, "r", encoding="utf-8") as context_file:
            messages = json.load(context_file)
        if not isinstance(messages, list):
            parser.error("The loaded context must be a JSON array of messages.")
        if args.query:
            messages.append({"role": "user", "content": args.query})
    else:
        if not args.query:
            parser.error("query is required unless --load is provided.")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful agent. Use available tools to answer user questions.",
            },
            {
                "role": "user",
                "content": args.query,
            },
        ]

    tool_calls = None
    for i in range(args.max_iterations):
        print(f"\nIteration {i + 1}")
        print("=" * 30)

        message = get_completion(messages, model="gpt-oss-120b")
        messages.append(message)
        tool_calls = message_field(message, "tool_calls")

        if not tool_calls:
            print(f"Agent Answer: {message.content}")
            break
        messages = execute_tool_calls(messages)

    if tool_calls:
        print("Agent did not find the answer within the iteration limit")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as context_file:
            json.dump([message_to_dict(message) for message in messages], context_file, indent=2)
            context_file.write("\n")


if __name__ == "__main__":
    main()
